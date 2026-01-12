import json
from functools import lru_cache


@lru_cache(maxsize=8)
def get_prompts(count=100):
    print("Loading dataset from ModelScope...")
    try:
        from modelscope.msdatasets import MsDataset

        ds = MsDataset.load(
            "AI-ModelScope/Chinese-DeepSeek-V3.2-Exp-chat-example", split="train"
        )
    except ImportError:
        print("Error: modelscope is not installed. Please install it.")
        return []
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return []

    print("Processing dataset...")

    candidates = []

    try:
        for item in ds:
            try:
                # Get prompt length, default to 0 if not present
                # The requirement is 32 ~ 64 tokens
                prompt_len = item.get("len_problem", 0)
                if 32 <= prompt_len <= 64:
                    candidates.append(item)
            except Exception:
                # Some items might be malformed or missing fields
                continue
    except Exception as e:
        print(f"Error during dataset iteration: {e}")
        return []

    print(
        f"Found {len(candidates)} conversations with prompt length between 32 and 64."
    )

    # Use a fixed sort to ensure deterministic results
    # Sorting by 'id' if available, otherwise by the JSON string representation
    candidates.sort(key=lambda x: x.get("id", json.dumps(x, sort_keys=True)))

    # Select the first 'count' conversations
    selected = candidates[:count]

    print(f"Selected {len(selected)} conversations.")

    # Return list of prompt strings
    return [item.get("problem", "") for item in selected]


if __name__ == "__main__":
    prompts = get_prompts()
    print(f"Got {len(prompts)} prompts")
