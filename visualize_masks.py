# ruff: noqa: E402
import logging
from pathlib import Path

# load environment variables
from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())

import hydra
from datasets import load_dataset
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from rich.console import Console
from rich.text import Text
from rich.panel import Panel

import explainable_medical_coding.config.factories as factories
from explainable_medical_coding.utils.loaders import (
    load_and_prepare_dataset,
    load_trained_model,
)
from explainable_medical_coding.utils.tokenizer import TargetTokenizer
from explainable_medical_coding.utils.analysis import calculate_selected_mask_ids
from explainable_medical_coding.utils.data_helper_functions import (
    create_targets_column,
    filter_unknown_targets,
    format_evidence_spans,
    get_unique_targets,
)
from explainable_medical_coding.utils.seed import set_seed
from explainable_medical_coding.utils.settings import TARGET_COLUMN, TEXT_COLUMN
from explainable_medical_coding.utils.tensor import deterministic, set_gpu
from explainable_medical_coding.eval.plausibility_metrics import compute_explanation_decision_boundary_from_reference

LOGGER = logging.getLogger(name=__file__)
LOGGER.setLevel(logging.INFO)

console = Console()

def load_and_prepare_reference_model(cfg, reference_model_path, text_tokenizer, autoregressive, target_columns, max_input_length, device):
    """Same function as in train_plm.py"""
    LOGGER.info(f"Reference model path: {reference_model_path}")
    saved_config = OmegaConf.load(reference_model_path / "config.yaml")
    reference_model, reference_decision_boundary = load_trained_model(
        reference_model_path,
        saved_config,
        pad_token_id=text_tokenizer.pad_token_id,
        device=device,
    )
    reference_model.to(device)
    reference_model.eval()
    LOGGER.info("Loading Reference Target Tokenizer from reference_model_path")
    reference_target_tokenizer = TargetTokenizer(autoregressive=autoregressive)
    reference_target_tokenizer.load(reference_model_path / "target_tokenizer.json")
    
    explainability_method = cfg.loss.configs.get('explanation_method', None)
    if explainability_method is None:
        raise ValueError("Explainability method must be specified for reference model")

    mdace_path = 'explainable_medical_coding/datasets/mdace_inpatient_icd9.py'
    mdace_dataset = load_and_prepare_dataset(
        mdace_path, 
        text_tokenizer, 
        reference_target_tokenizer, 
        max_input_length, 
        target_columns
    )
    explanation_decision_boundary, explainer_callable = compute_explanation_decision_boundary_from_reference(
        model=reference_model,
        model_path=reference_model_path,
        explainability_method=explainability_method,
        dataset=mdace_dataset,
        text_tokenizer=text_tokenizer,
        target_tokenizer=reference_target_tokenizer,
        cache_explanations=False,
        decision_boundary=reference_decision_boundary,
    )

    return reference_model, reference_decision_boundary, explanation_decision_boundary, explainer_callable

def highlight_tokens_in_text(text: str, input_ids: list, tokenizer: AutoTokenizer, selected_mask_ids: list) -> Text:
    """Create highlighted text visualization"""
    rich_text = Text()
    
    for i, token_id in enumerate(input_ids):
        token_text = tokenizer.decode([token_id], skip_special_tokens=False)
        
        if i in selected_mask_ids:
            rich_text.append(token_text, style="bold white on red")
        else:
            rich_text.append(token_text, style="default")
    
    return rich_text

@hydra.main(
    version_base=None,
    config_path="explainable_medical_coding/config",
    config_name="config",
)
def visualize_masks(cfg: OmegaConf) -> None:
    """Visualization function using the same logic as train_plm.py"""
    
    console.print("[bold green]Starting mask visualization...[/bold green]")
    
    if cfg.deterministic:
        deterministic()

    set_seed(cfg.seed)
    device = set_gpu(cfg)

    target_columns = list(cfg.data.target_columns)
    dataset_path = Path(cfg.data.dataset_path)
    reference_model_path = Path(cfg.loss.configs.reference_model_path)
    
    # Load dataset (same as train_plm.py)
    dataset = load_dataset(str(dataset_path), trust_remote_code=True)

    text_tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.configs.model_path,
    )
    max_input_length = int(cfg.data.max_length)
    
    # Tokenize text
    dataset = dataset.map(
        lambda x: text_tokenizer(
            x[TEXT_COLUMN],
            return_length=True,
            truncation=True,
            max_length=max_input_length,
        ),
        batched=True,
        num_proc=8,
        batch_size=1_000,
        desc="Tokenizing text",
    )

    dataset = dataset.map(
        lambda x: create_targets_column(x, target_columns),
        desc="Creating targets column",
    )
    known_targets = set(get_unique_targets(dataset))
    dataset = dataset.map(
        lambda x: filter_unknown_targets(x, known_targets=known_targets),
        desc="Filter unknown targets",
    )
    dataset = dataset.filter(
        lambda x: len(x[TARGET_COLUMN]) > 0, desc="Filtering empty targets"
    )
    dataset = dataset.map(lambda x: format_evidence_spans(x, text_tokenizer))

    autoregressive = bool(cfg.model.autoregressive)
    target_tokenizer = TargetTokenizer(autoregressive=autoregressive)
    
    unique_targets = get_unique_targets(dataset)
    target_tokenizer.fit(unique_targets)

    # Convert targets to target ids
    dataset = dataset.map(
        lambda x: {"target_ids": target_tokenizer(x[TARGET_COLUMN])},
        desc="Converting targets to target ids",
    )
    dataset.set_format(
        type="torch", columns=["input_ids", "length", "attention_mask", "target_ids", "text", "target"]
    )

    # Load reference model and prepare explainer (same as train_plm.py)
    console.print("[bold blue]Loading reference model...[/bold blue]")
    reference_model, reference_decision_boundary, explanation_decision_boundary, explainer_callable = load_and_prepare_reference_model(
        cfg,
        reference_model_path,
        text_tokenizer,
        autoregressive,
        target_columns,
        max_input_length,
        device
    )

    console.print(f"[bold magenta]Reference decision boundary: {reference_decision_boundary:.4f}[/bold magenta]")
    console.print(f"[bold magenta]Explanation decision boundary: {explanation_decision_boundary:.4f}[/bold magenta]")

    # Take a small sample for visualization
    sample_size = 6  # Increased to test batching
    sample_dataset = dataset["train"].select(range(sample_size))
    
    console.print(f"[bold cyan]Processing {sample_size} samples in batches...[/bold cyan]")

    # Test the batched version of calculate_selected_mask_ids
    console.print("[bold blue]Testing batched processing...[/bold blue]")
    
    # Apply calculate_selected_mask_ids in batched mode
    batch_size = 3  # Process 3 samples at a time
    batched_results = sample_dataset.map(
        lambda x: calculate_selected_mask_ids(
            explainer_callable=explainer_callable,
            reference_model=reference_model,
            x=x,
            explanation_decision_boundary=explanation_decision_boundary,
            device=device,
            decision_boundary=reference_decision_boundary,
            stride=3
        ),
        desc="Testing batched mask calculation",
        batched=True,
        batch_size=batch_size,
    )

    console.print(f"[bold green]Batched processing completed successfully![/bold green]")
    console.print(f"Processed {len(batched_results)} samples")

    # Visualize the results
    for i, sample in enumerate(batched_results):
        console.print(f"\n[bold yellow]Sample {i+1}:[/bold yellow]")
        
        # Get original text and target codes
        original_text = sample[TEXT_COLUMN]
        target_codes = sample[TARGET_COLUMN]
        input_ids = sample["input_ids"].tolist()
        selected_mask_ids = sample["selected_mask_ids"]
        
        console.print(f"Target codes: {target_codes}")
        console.print(f"Text length: {len(original_text)} chars, Token length: {len(input_ids)} tokens")
        console.print(f"Selected mask positions: {selected_mask_ids}")
        console.print(f"Number of highlighted tokens: {len(selected_mask_ids)}/{len(input_ids)} ({len(selected_mask_ids)/len(input_ids)*100:.1f}%)")
        
        try:
            # Create highlighted visualization
            highlighted_text = highlight_tokens_in_text(
                original_text, input_ids, text_tokenizer, selected_mask_ids
            )
            
            # Display results
            console.print(Panel(highlighted_text, title=f"Sample {i+1} - Highlighted Important Tokens (Batched)"))
            
            # Show some highlighted tokens
            highlighted_tokens = []
            for pos in selected_mask_ids[:10]:  # Show first 10 highlighted tokens
                if pos < len(input_ids):
                    token = text_tokenizer.decode([input_ids[pos]], skip_special_tokens=False)
                    highlighted_tokens.append(f"'{token.strip()}'")
            
            if highlighted_tokens:
                console.print(f"Sample highlighted tokens: {', '.join(highlighted_tokens)}")
            
        except Exception as e:
            console.print(f"[bold red]Error visualizing sample {i+1}: {e}[/bold red]")
            import traceback
            traceback.print_exc()

    # Test single example processing for comparison
    console.print(f"\n[bold blue]Testing single example processing for comparison...[/bold blue]")
    
    single_sample = sample_dataset[0]  # Get first sample
    console.print(f"\n[bold yellow]Single Sample Test:[/bold yellow]")
    
    try:
        # Calculate selected mask ids for single example
        single_result = calculate_selected_mask_ids(
            explainer_callable=explainer_callable,
            reference_model=reference_model,
            x=single_sample,
            explanation_decision_boundary=explanation_decision_boundary,
            device=device,
            decision_boundary=reference_decision_boundary,
            stride=3
        )
        
        single_selected_mask_ids = single_result["selected_mask_ids"]
        batched_selected_mask_ids = batched_results[0]["selected_mask_ids"]
        
        console.print(f"Single processing result: {single_selected_mask_ids}")
        console.print(f"Batched processing result: {batched_selected_mask_ids}")
        
        # Check if results match
        if single_selected_mask_ids == batched_selected_mask_ids:
            console.print("[bold green]✅ Single and batched processing results match![/bold green]")
        else:
            console.print("[bold red]❌ Single and batched processing results differ![/bold red]")
            console.print(f"Difference: {set(single_selected_mask_ids) ^ set(batched_selected_mask_ids)}")
        
    except Exception as e:
        console.print(f"[bold red]Error in single example test: {e}[/bold red]")
        import traceback
        traceback.print_exc()

    console.print("\n[bold green]Visualization completed![/bold green]")

if __name__ == "__main__":
    visualize_masks()