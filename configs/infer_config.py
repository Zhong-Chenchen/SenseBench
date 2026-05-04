MODEL_LIST = [
    # ── API 商用模型（需要 --api_key / OPENAI_API_KEY） ──
    # "api:gpt-4o",
    # "api:gpt-4o-mini",
    # "api:claude-3-5-sonnet-20241022",
    # "api:qwen-vl-max",
    # "api:qwen-vl-plus",
    # "api:gemini-1.5-pro",

    # ── 本地模型 ──
    # "Qwen/Qwen3-VL-32B-Instruct",
    # "ZhipuAI/glm-4v-9b",
    # "moonshotai/Kimi-VL-A3B-Instruct",
    # "OpenGVLab/InternVL3_5-38B",
    # "deepseek-ai/deepseek-vl-7b-chat",
    # "LLM-Research/Llama-3.2-11B-Vision-Instruct",
    # "rs:LHRS-Bot",
    # "rs:EarthDial",
    "AIDC-AI/Ovis2.5-9B",
    "deepseek-ai/deepseek-vl-7b-chat",
    "OpenBMB/MiniCPM-V-4_5",
    "rs_GeoChat",
    "rs:LHRS-Bot",
    "rs:EarthDial"
]

# ── Single bench 完整任务表 ──

_OPTICAL_WHETHER_WHAT = [
    "blur_gaussian", "blur_motion",
    "cloud_haze", "cloud_real",
    "compression_jpeg2000", "compression_jpeg", "compression_SPIHT", "compression_webp",
    "correction_color_bandattenuation", "correction_color_bandswitch",
    "correction_compression", "correction_stretching",
    "missing_blind_flickering", "missing_linerdeadpixels", "missing_tilesmiss",
    "noise_deadline", "noise_gaussian", "noise_impulse",
    "noise_spatially_correlated", "noise_stripe",
]

_OPTICAL_HOW = [
    "blur_gaussian", "blur_motion",
    "cloud_haze", "cloud_real",
    "compression_jpeg2000", "compression_jpeg", "compression_SPIHT", "compression_webp",
    "noise_deadline", "noise_gaussian", "noise_impulse",
    "noise_spatially_correlated", "noise_stripe",
]

_OPTICAL_DESCRIPTION_SINGLE = [
    "blur_gaussian", "blur_motion",
    "cloud_haze", "cloud_simplex",
    "compression_jpeg2000", "compression_jpeg", "compression_SPIHT", "compression_webp",
    "correction_color_bandattenuation", "correction_color_bandswitch",
    "correction_compression", "correction_stretching",
    "missing_blind_flickering", "missing_linerdeadpixels", "missing_tilesmiss",
    "noise_deadline", "noise_gaussian", "noise_impulse",
    "noise_spatially_correlated", "noise_stripe",
]

RSBENCH_TASKS = {
    "whether": {
        "optical": _OPTICAL_WHETHER_WHAT,
        "sar":     ["Speckle", "Sidelobe"],
    },
    "what": {
        "optical": _OPTICAL_WHETHER_WHAT,
        "sar":     ["Speckle", "Sidelobe"],
        "multi":   ["multi_distortion"],
    },
    "how": {
        "optical": _OPTICAL_HOW,
    },
    "description": {
        "optical": _OPTICAL_DESCRIPTION_SINGLE,
    },
}

# ── Multi bench 完整任务表 ──

_MULTI_TASKS = [
    "blur_gaussian", "blur_motion",
    "cloud_haze", "cloud_simplex",
    "compression_jpeg2000", "compression_jpeg", "compression_SPIHT", "compression_webp",
    "correction_color_bandattenuation", "correction_color_bandswitch",
    "correction_compression", "correction_stretching",
    "missing_blind_flickering", "missing_linerdeadpixels", "missing_tilesmiss",
    "noise_deadline", "noise_gaussian", "noise_impulse",
    "noise_spatially_correlated", "noise_stripe",
]

RSBENCH_TASKS_MULTI = {
    "whether":     {"optical": _MULTI_TASKS},
    "what":        {"optical": _MULTI_TASKS},
    "how":         {"optical": _MULTI_TASKS},
    "description": {"optical": _MULTI_TASKS},
}


sys_prompt_4choice = """Please select the most appropriate answer for the following single-choice question from the given options. Only respond with the corresponding letter (A, B, C or D). Do not include any additional text.
"""
sys_prompt_3choice = """Please select the most appropriate answer for the following single-choice question from the given options. Only respond with the corresponding letter (A, B, C). Do not include any additional text.
"""
sys_prompt_2choice = """Please select the most appropriate answer for the following single-choice question from the given options. Only respond with the corresponding letter (A or B). Do not include any additional text.
"""
sys_prompt_multi = """Please select the two correct answers for the following question from the given options. Respond only with the two letters in alphabetical order, separated by a comma with no space (e.g., A,B). Do not include any additional text."""
