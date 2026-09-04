"""Helpers for searchable Gradio object selector lists."""


def normalize_object_selector_choices(choices):
    """Return stable, non-empty object names from Gradio choices or raw lists."""
    normalized = []
    seen = set()
    for choice in choices or []:
        value = choice
        if isinstance(choice, (list, tuple)) and len(choice) >= 2:
            value = choice[1]
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        normalized.append(text)
    return normalized


def filter_object_selector_choices(choices, query):
    object_names = normalize_object_selector_choices(choices)
    keyword = str(query or "").strip().casefold()
    if not keyword:
        return object_names
    return [name for name in object_names if keyword in name.casefold()]


def visible_object_selector_value(visible_choices, selected_choices):
    selected = set(normalize_object_selector_choices(selected_choices))
    return [
        name
        for name in normalize_object_selector_choices(visible_choices)
        if name in selected
    ]


def merge_object_selector_selection(
    previous_selected,
    visible_choices,
    visible_selected,
):
    visible = normalize_object_selector_choices(visible_choices)
    visible_set = set(visible)
    visible_selected_set = set(normalize_object_selector_choices(visible_selected))
    retained = [
        name
        for name in normalize_object_selector_choices(previous_selected)
        if name not in visible_set
    ]
    return normalize_object_selector_choices(
        retained + [name for name in visible if name in visible_selected_set]
    )


def effective_object_selector_selection(
    current_visible,
    visible_choices,
    selected_state,
):
    return merge_object_selector_selection(
        selected_state,
        visible_choices,
        current_visible,
    )
