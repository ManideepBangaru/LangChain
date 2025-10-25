import yaml
from pathlib import Path
from typing import Any, Dict
import os


def read_yaml(file_path: str | Path) -> Dict[str, Any]:
    """
    Read and parse a YAML file.
    
    Args:
        file_path: Path to the YAML file to read
        
    Returns:
        Dictionary containing the parsed YAML content
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        yaml.YAMLError: If the file contains invalid YAML
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {file_path}")
    
    with open(path, 'r', encoding='utf-8') as file:
        try:
            content = yaml.safe_load(file)
            return content if content is not None else {}
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file {file_path}: {e}")


def load_prompt(prompt_name: str) -> Dict[str, Any]:
    """
    Load a YAML prompt file from the prompts folder.
    
    Args:
        prompt_name: Name of the prompt file (with or without .yaml extension)
        
    Returns:
        Dictionary containing the parsed YAML content
        
    Raises:
        FileNotFoundError: If the prompt file doesn't exist
        yaml.YAMLError: If the file contains invalid YAML
    """
    # Get the project root directory (where this script is located)
    current_dir = Path(__file__).parent.parent
    prompts_dir = current_dir / "prompts"
    
    # Add .yaml extension if not present
    if not prompt_name.endswith('.yaml') and not prompt_name.endswith('.yml'):
        prompt_name += '.yaml'
    
    prompt_path = prompts_dir / prompt_name
    return read_yaml(prompt_path)
