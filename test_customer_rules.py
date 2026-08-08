from classify_fresh_food_relevance import build_identity_prompt
from src.harmonize.customer_rules import (
    apply_customer_category_overrides,
)


def test_customer_category_overrides_assign_requested_categories():
    assert apply_customer_category_overrides("sonstiges", product_name="Erbsen tiefgefroren") == "tk"
    assert apply_customer_category_overrides("fisch", product_name="Friesenkrone Heringssalat") == "mopro"
    assert apply_customer_category_overrides("fleisch", product_name="Delikatess Bacon") == "wurst"
    assert apply_customer_category_overrides("sonstiges", product_name="Rostbratwürste") == "wurst"


def test_customer_category_overrides_move_glass_and_can_to_other():
    assert apply_customer_category_overrides("wurst", product_name="Hot Dog Würstchen", unit="glas") == "sonstiges"
    assert apply_customer_category_overrides("mopro", product_name="Kondensmilch Dose") == "sonstiges"


def test_identity_prompt_mentions_customer_feedback_negative_examples():
    prompt = build_identity_prompt({"category": "obst_gemuese", "product_name": "Guacamole"})

    assert "Delikatess-Sahne-Heringsfilets => hard exclusion, Prepared Fish" in prompt
    assert "Pizzasauce => hard exclusion, Sauce" in prompt
    assert "Guacamole => hard exclusion, Dip" in prompt
    assert "Gewürzgurken => hard exclusion, Preserved" in prompt
    assert "Toilettenpapier => hard exclusion, Non Food" in prompt
    assert "Iglo Gemüse or ja! Milch => hard exclusion, Blocked brand" in prompt
    assert "Matjesfilets => hard exclusion, Preserved" in prompt
    assert "Nordseekrabbensalat or Sahne-Heringsfilets => hard exclusion, Prepared Fish" in prompt
    assert "Frikadellen gebraten or gekochte Garnelen => hard exclusion, Prepared" in prompt
    assert "Schweinelachse => raw pork product, not fish" in prompt
