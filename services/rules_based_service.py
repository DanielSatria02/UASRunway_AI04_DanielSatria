"""
Rules-based inference service for product prioritization and recommendations.
"""
from typing import Dict, Tuple

import pandas as pd


def categorize_for_success(row: pd.Series) -> str:
    """Recategorize products into weather-appropriate collection groups."""
    master_category = str(row.get("masterCategory", "")).strip()
    season = str(row.get("season", "")).strip()

    if master_category == "Accessories":
        return "Sunny Days Collections"
    if season in ["Winter", "Fall"] and master_category == "Apparel":
        return "Rainy Day Attires"
    if season in ["Spring", "Summer"] and master_category == "Apparel":
        return "Sunny Days Collections"
    return "Other/Not Categorized"


def calculate_product_priority_score(row: pd.Series) -> Tuple[int, str]:
    """Compute a priority score and human-readable reasons for each product."""
    score = 0
    reasons = []

    season = str(row.get("season", "")).strip()
    master_category = str(row.get("masterCategory", "")).strip()

    if season in ["Winter", "Fall"]:
        score += 50
        reasons.append("Product from unsuitable season.")
    elif season in ["Spring", "Summer"]:
        score += 10
        reasons.append("Product from suitable season.")
    else:
        score += 5
        reasons.append("Unspecified season.")

    if master_category == "Apparel" and season in ["Winter", "Fall"]:
        score += 30
        reasons.append("Apparel from unsuitable season poses a significant inventory challenge.")
    elif master_category == "Accessories" and season in ["Winter", "Fall"]:
        score += 15
        reasons.append("Accessories from unsuitable season.")
    elif master_category == "Footwear" and season in ["Winter", "Fall"]:
        score += 15
        reasons.append("Footwear from unsuitable season.")
    elif master_category == "Personal Care" and season in ["Winter", "Fall"]:
        score += 5
        reasons.append("Personal Care items might have less seasonal impact.")

    return score, "; ".join(reasons) if reasons else "No specific priority rule applied."


def label_ai_product_priority(score: int) -> str:
    """Convert score into a priority label."""
    if score >= 80:
        return "High"
    if score >= 30:
        return "Medium"
    return "Low"


def get_knowledge_base_rules() -> Dict[str, Dict]:
    """Return the rules dictionary used by the inference engine."""
    return {
        "rule_high_priority_rainy_attire": {
            "condition": lambda row: row["ai_product_priority"] == "High"
            and row["ai_success_category"] == "Rainy Day Attires",
            "recommendation": "Launch targeted marketing campaign for Rainy Day Attires. Bundle with complementary items.",
            "escalation": "Consider cross-promotional events with local stores or influencers.",
            "reason_template": "High priority apparel from unsuitable original season, now recategorized for Rainy Days. Focus on local seasonal appeal.",
        },
        "rule_high_priority_sunny_collections": {
            "condition": lambda row: row["ai_product_priority"] == "High"
            and row["ai_success_category"] == "Sunny Days Collections",
            "recommendation": "Promote as 'Summer Essentials' or 'Tropical Getaway Must-Haves'. Utilize online channels.",
            "escalation": "Evaluate inventory levels for flash sales if stock is high.",
            "reason_template": "High priority accessories from unsuitable original season, now recategorized for Sunny Days. Emphasize versatility and fashion.",
        },
        "rule_medium_priority_rainy_attire": {
            "condition": lambda row: row["ai_product_priority"] == "Medium"
            and row["ai_success_category"] == "Rainy Day Attires",
            "recommendation": "Include in general 'Rainy Season' promotions. Review for minor re-styling if cost-effective.",
            "escalation": "Monitor sales performance closely; consider pricing adjustments if slow.",
            "reason_template": "Medium priority item recategorized for Rainy Days. Integrate into broader seasonal strategy.",
        },
        "rule_medium_priority_sunny_collections": {
            "condition": lambda row: row["ai_product_priority"] == "Medium"
            and row["ai_success_category"] == "Sunny Days Collections",
            "recommendation": "Feature in general 'Sunny Season' displays. Explore pairing with other best-selling items.",
            "escalation": "Gather customer feedback on product appeal for potential adjustments.",
            "reason_template": "Medium priority item recategorized for Sunny Days. Optimize placement and bundling.",
        },
        "rule_low_priority_any_category": {
            "condition": lambda row: row["ai_product_priority"] == "Low",
            "recommendation": "Maintain standard marketing and sales channels. No immediate special action.",
            "escalation": "No immediate escalation needed, normal inventory management.",
            "reason_template": "Low priority item. Continue with standard operational procedures.",
        },
        "rule_other_not_categorized": {
            "condition": lambda row: row["ai_success_category"] == "Other/Not Categorized",
            "recommendation": "Conduct detailed individual product review. Assess potential for repurposing or niche market targeting.",
            "escalation": "Escalate to product development for redesign consideration or potential liquidation analysis.",
            "reason_template": "Product could not be easily recategorized. Requires manual expert review for specific action.",
        },
    }


def run_inference_engine_for_df_row(row: pd.Series, rules: Dict[str, Dict]) -> Tuple[str, str, str]:
    """Run rule matching for a single row and return recommendation fields."""
    recommendation = "No specific recommendation found."
    escalation = "No specific escalation."
    reason = "No rule matched the current facts."

    for rule_details in rules.values():
        if rule_details["condition"](row):
            recommendation = rule_details["recommendation"]
            escalation = rule_details["escalation"]
            reason = rule_details["reason_template"]
            break

    return recommendation, escalation, reason


def apply_rules_based_inference(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply recategorization, priority scoring, and rule-based inference to DataFrame.

    Returns a new DataFrame with inferred rule-based columns.
    """
    result_df = df.copy()

    result_df["ai_success_category"] = result_df.apply(categorize_for_success, axis=1)

    score_results = result_df.apply(calculate_product_priority_score, axis=1)
    result_df["product_priority_score"] = score_results.apply(lambda x: x[0])
    result_df["product_priority_reasons"] = score_results.apply(lambda x: x[1])
    result_df["ai_product_priority"] = result_df["product_priority_score"].apply(label_ai_product_priority)

    rules = get_knowledge_base_rules()
    result_df[["expert_recommendation", "expert_escalation", "expert_reason"]] = result_df.apply(
        lambda row: pd.Series(run_inference_engine_for_df_row(row, rules)),
        axis=1,
    )

    return result_df


def build_rules_based_output_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build a compact, display-ready table for the Streamlit dashboard."""
    inferred_df = apply_rules_based_inference(df)
    return inferred_df[
        [
            "productDisplayName",
            "season",
            "masterCategory",
            "ai_product_priority",
            "ai_success_category",
            "expert_recommendation",
            "expert_escalation",
            "expert_reason",
        ]
    ]