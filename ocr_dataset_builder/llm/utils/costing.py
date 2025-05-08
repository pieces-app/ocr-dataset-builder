import logging

# Cost calculation utility
# Define pricing directly here for clarity and up-to-date values
MODEL_PRICING = {
    # Gemini 2.5 Pro (Preview - May 2025)
    "gemini-2.5-pro-preview-05-06": {
        "threshold_k": 200,
        "<=200k": {"input": 1.25, "output": 10.00},
        ">200k": {"input": 2.50, "output": 15.00},
    },
    # Alias for previous 2.5 Pro pointing to the new one
    "gemini-2.5-pro-preview-03-25": {
        "threshold_k": 200,
        "<=200k": {"input": 1.25, "output": 10.00},
        ">200k": {"input": 2.50, "output": 15.00},
    },
    # Gemini 1.5 Pro
    "gemini-1.5-pro-latest": {
        "threshold_k": 128,
        "<=128k": {"input": 1.25, "output": 5.00},
        ">128k": {"input": 2.50, "output": 10.00},
    },
    # Gemini 1.5 Flash
    "gemini-1.5-flash-latest": {
        "threshold_k": 128,
        "<=128k": {"input": 0.075, "output": 0.30},
        ">128k": {"input": 0.15, "output": 0.60},
    },
    # Gemini 2.0 Flash (Assuming text rates, no threshold mentioned)
    "gemini-2.0-flash-latest": {
        "input": 0.10,
        "output": 0.40,
    },
    # Gemini 2.0 Flash-Lite (No threshold mentioned)
    "gemini-2.0-flash-lite-latest": {
        "input": 0.075,
        "output": 0.30,
    },
    # Gemini 2.5 Flash (Preview - May 2025)
    "gemini-2.5-flash-preview-05-07": { # Assuming model name based on pattern
        "input": 0.15, # Text rate
        "output": 0.60, # Non-thinking rate
    },
    # Gemini 1.5 Flash-8B
    "gemini-1.5-flash-8b-latest": { # Assuming model name pattern
         "threshold_k": 128,
        "<=128k": {"input": 0.0375, "output": 0.15},
        ">128k": {"input": 0.075, "output": 0.30},
    },
    # Add other models here if needed, e.g., specific versions like -001
    # Note: Pricing for gemini-pro / gemini-pro-vision was not found in the provided text.
}

def calculate_gemini_cost(
    model_name: str,
    input_tokens: int | None,
    output_tokens: int | None,
) -> float:
    """
    Calculates the estimated cost for a Gemini API call based on token counts.
    Uses the MODEL_PRICING dictionary defined above.

    Args:
        model_name: The name of the Gemini model variant used.
        input_tokens: The number of input tokens.
        output_tokens: The number of output tokens.

    Returns:
        The estimated cost in USD, or 0.0 if pricing is not defined or tokens are missing.
    """
    if input_tokens is None or output_tokens is None:
        logging.warning(f"Cannot calculate cost for '{model_name}' due to missing token counts.")
        return 0.0

    # Attempt to find pricing for the specific model or a base model
    pricing_config = MODEL_PRICING.get(model_name)
    if not pricing_config:
        # Try base names if a versioned name wasn't found
        base_model_name = None
        if "-pro" in model_name: # Simplistic check, might need refinement
            base_model_name = "gemini-1.5-pro-latest" # Default to 1.5 Pro if unsure
            if "2.5" in model_name:
                 base_model_name = "gemini-2.5-pro-preview-05-06"
        elif "-flash" in model_name:
             base_model_name = "gemini-1.5-flash-latest"
             if "2.0" in model_name:
                 base_model_name = "gemini-2.0-flash-latest"
        
        if base_model_name:
            pricing_config = MODEL_PRICING.get(base_model_name)
            if pricing_config:
                logging.debug(f"Pricing not found for '{model_name}', using base '{base_model_name}' rates.")
            else:
                 logging.warning(f"Pricing config not found for model '{model_name}' or its likely base '{base_model_name}'. Cost calculation will be 0.0")
                 return 0.0
        else:
             logging.warning(f"Pricing config not found for model '{model_name}'. Cost calculation will be 0.0")
             return 0.0

    threshold_k = pricing_config.get("threshold_k")
    input_rate = 0.0
    output_rate = 0.0

    if threshold_k:
        threshold_tokens = threshold_k * 1000
        tier_key_low = f"<={threshold_k}k"
        tier_key_high = f">{threshold_k}k"
        
        # Determine which tier applies based on input tokens
        if input_tokens <= threshold_tokens:
            tier_rates = pricing_config.get(tier_key_low)
            if tier_rates:
                input_rate = tier_rates.get("input", 0.0)
                output_rate = tier_rates.get("output", 0.0)
                tier_used = tier_key_low
            else:
                 logging.warning(f"Tier rates '{tier_key_low}' not found for '{model_name}'. Cost is 0.0")
                 return 0.0
        else: # Input tokens > threshold
            tier_rates = pricing_config.get(tier_key_high)
            if tier_rates:
                input_rate = tier_rates.get("input", 0.0)
                output_rate = tier_rates.get("output", 0.0)
                tier_used = tier_key_high
            else:
                 logging.warning(f"Tier rates '{tier_key_high}' not found for '{model_name}'. Cost is 0.0")
                 return 0.0
        log_msg_suffix = f"(Tier: {tier_used})"
    else:
        # No threshold, use base rates
        input_rate = pricing_config.get("input", 0.0)
        output_rate = pricing_config.get("output", 0.0)
        log_msg_suffix = "(Single Tier)"
        if input_rate == 0.0 and output_rate == 0.0:
             logging.warning(f"Base input/output rates not found for '{model_name}'. Cost is 0.0")
             return 0.0


    input_cost = (input_tokens / 1_000_000) * input_rate
    output_cost = (output_tokens / 1_000_000) * output_rate
    total_cost = input_cost + output_cost

    logging.debug(
        f"Cost calc ({model_name} {log_msg_suffix}): "
        f"In:{input_tokens}tk @ ${input_rate:.4f}/M = ${input_cost:.6f}, "
        f"Out:{output_tokens}tk @ ${output_rate:.4f}/M = ${output_cost:.6f}, "
        f"Total=${total_cost:.6f}"
    )
    return total_cost 