# Two-Room Memory Architecture
# Efficient LLM memory management via triviality gating

from .classifier_gate import (
    process_exchange,
    predict,
    should_persist,
    persist,
    get_room2_contents,
    clear_room2,
)

__version__ = "0.1.0"
__author__ = "Zachary Epstein and Claude (Anthropic)"
