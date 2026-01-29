"""
Brand color palettes for OCR vs VLM evaluation visualizations.

Consistent colors for models (by provider) and datasets.
"""

from typing import Dict, Optional

# =============================================================================
# PROVIDER COLOR PALETTES
# =============================================================================

AZURE = {
    'primary': '#0078D4',      # Core Azure blue
    'dark': '#005A9E',         # Darker blue for contrast
    'light': '#2B88D8',        # Lighter accent blue
    'background': '#E5F1FB',   # Very light blue background
    'text': '#1F1F1F',         # Neutral dark for text pairing
}

MISTRAL = {
    'primary': '#F04E23',      # Signature orange
    'dark': '#C63D1E',         # Deeper burnt orange
    'light': '#FF8A65',        # Soft orange highlight
    'background': '#F6F6F6',   # Off-white background
    'text': '#2A2A2A',         # Charcoal black
}

OPENAI = {
    'primary': '#10A37F',      # Primary green
    'dark': '#0E8F6E',         # Darker green
    'light': '#6EE7C8',        # Light mint accent
    'background': '#F7F7F8',   # Light neutral background
    'text': '#202123',         # Near-black UI base
}

CLAUDE = {
    'primary': '#B85C38',      # Warm brown-orange
    'dark': '#8F4326',         # Darker earth tone
    'light': '#E2B6A0',        # Soft beige accent
    'background': '#FAF7F5',   # Warm off-white
    'text': '#2B2B2B',         # Dark neutral
}

# =============================================================================
# MODEL COLORS (mapped to providers)
# =============================================================================

MODEL_COLORS: Dict[str, str] = {
    # Azure models
    'azure_intelligence': AZURE['primary'],
    'azure_doc_intel': AZURE['primary'],
    
    # Mistral models
    'mistral_document_ai': MISTRAL['primary'],
    'mistral_ocr': MISTRAL['primary'],
    
    # OpenAI models
    'gpt-5-mini': OPENAI['primary'],
    'gpt-5-nano': OPENAI['dark'],
    'gpt-4o-mini': OPENAI['light'],
    
    # Claude models
    'claude_sonnet': CLAUDE['primary'],
    'claude_haiku': CLAUDE['dark'],
    
    # Open-source models (neutral gray tones)
    'donut': '#6B7280',
    'qwen_vl': '#9CA3AF',
    'deepseek_ocr': '#4B5563',
}

# =============================================================================
# DATASET COLORS
# =============================================================================

DATASET_COLORS: Dict[str, str] = {
    # QA Datasets (blue-green spectrum)
    'DocVQA_mini': '#3B82F6',       # Blue
    'InfographicVQA_mini': '#06B6D4', # Cyan
    
    # Parsing Datasets (purple-pink spectrum)
    'IAM_mini': '#8B5CF6',          # Purple
    'ICDAR_mini': '#EC4899',        # Pink
    
    # Specialized Datasets (warm tones)
    'VOC2007': '#F59E0B',           # Amber
    'publaynet': '#EF4444',         # Red
    'publaynet_full': '#DC2626',    # Dark red
}

# =============================================================================
# APPROACH COLORS
# =============================================================================

APPROACH_COLORS: Dict[str, str] = {
    'ocr_pipeline': AZURE['primary'],      # OCR → LLM
    'vlm_pipeline': OPENAI['primary'],     # VLM → LLM
    'direct_vqa': CLAUDE['primary'],       # VLM only
    'preextracted': MISTRAL['primary'],    # External OCR → LLM
}

# =============================================================================
# COMBINED COLORS DICT
# =============================================================================

COLORS = {
    'azure': AZURE,
    'mistral': MISTRAL,
    'openai': OPENAI,
    'claude': CLAUDE,
    'models': MODEL_COLORS,
    'datasets': DATASET_COLORS,
    'approaches': APPROACH_COLORS,
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_model_color(model_name: str, shade: str = 'primary') -> str:
    """
    Get color for a model by name.
    
    Args:
        model_name: Model identifier (e.g., 'gpt-5-mini', 'azure_intelligence')
        shade: 'primary', 'dark', or 'light'
    
    Returns:
        Hex color string
    """
    # Direct lookup
    if model_name in MODEL_COLORS:
        base_color = MODEL_COLORS[model_name]
        if shade == 'primary':
            return base_color
        # Find provider for shade variants
        for provider_name, provider in [('azure', AZURE), ('mistral', MISTRAL), 
                                         ('openai', OPENAI), ('claude', CLAUDE)]:
            if base_color == provider['primary']:
                return provider.get(shade, base_color)
        return base_color
    
    # Fallback: detect provider from name
    name_lower = model_name.lower()
    if 'azure' in name_lower:
        return AZURE.get(shade, AZURE['primary'])
    elif 'mistral' in name_lower:
        return MISTRAL.get(shade, MISTRAL['primary'])
    elif 'gpt' in name_lower or 'openai' in name_lower:
        return OPENAI.get(shade, OPENAI['primary'])
    elif 'claude' in name_lower:
        return CLAUDE.get(shade, CLAUDE['primary'])
    
    # Default gray
    return '#6B7280'


def get_dataset_color(dataset_name: str) -> str:
    """
    Get color for a dataset by name.
    
    Args:
        dataset_name: Dataset identifier (e.g., 'DocVQA_mini', 'IAM_mini')
    
    Returns:
        Hex color string
    """
    # Direct lookup
    if dataset_name in DATASET_COLORS:
        return DATASET_COLORS[dataset_name]
    
    # Partial match
    name_lower = dataset_name.lower()
    for key, color in DATASET_COLORS.items():
        if key.lower() in name_lower or name_lower in key.lower():
            return color
    
    # Default gray
    return '#6B7280'


def get_approach_color(approach: str) -> str:
    """
    Get color for an experimental approach.
    
    Args:
        approach: Approach name ('ocr_pipeline', 'vlm_pipeline', 'direct_vqa', 'preextracted')
    
    Returns:
        Hex color string
    """
    return APPROACH_COLORS.get(approach, '#6B7280')


def create_color_scale(base_color: str, n_shades: int = 5) -> list:
    """
    Create a gradient scale from a base color.
    
    Args:
        base_color: Hex color string
        n_shades: Number of shades to generate
    
    Returns:
        List of hex color strings from light to dark
    """
    import colorsys
    
    # Parse hex to RGB
    hex_color = base_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) / 255 for i in (0, 2, 4))
    
    # Convert to HLS
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    
    # Generate shades
    shades = []
    for i in range(n_shades):
        # Vary lightness from 0.9 (light) to 0.3 (dark)
        new_l = 0.9 - (i * 0.6 / (n_shades - 1)) if n_shades > 1 else l
        new_r, new_g, new_b = colorsys.hls_to_rgb(h, new_l, s)
        hex_shade = '#{:02x}{:02x}{:02x}'.format(
            int(new_r * 255), int(new_g * 255), int(new_b * 255)
        )
        shades.append(hex_shade)
    
    return shades
