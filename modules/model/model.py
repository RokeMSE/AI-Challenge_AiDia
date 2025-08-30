from modules.modeling import CLIP4Clip

model = CLIP4Clip.from_pretrained(
    cross_model_name="cross-base",
    cache_dir="",
    state_dict=None,
    task_config={}
)