from classify_fresh_food_relevance import build_identity_prompt
from src.harmonize.customer_rules import (
    apply_customer_category_overrides,
    apply_customer_category_overrides_to_product,
    apply_customer_category_overrides_to_row,
)
from src.models import RawProduct


def test_customer_category_api_does_not_infer_from_product_words():
    assert apply_customer_category_overrides("sonstiges", product_name="Erbsen tiefgefroren") == "sonstiges"
    assert apply_customer_category_overrides("fisch", product_name="Friesenkrone Heringssalat") == "fisch"
    assert apply_customer_category_overrides("fleisch", product_name="Delikatess Bacon") == "fleisch"
    assert apply_customer_category_overrides("sonstiges", product_name="Rostbratwürste") == "sonstiges"


def test_customer_category_api_does_not_infer_from_packaging():
    assert apply_customer_category_overrides("wurst", product_name="Hot Dog Würstchen", unit="glas") == "wurst"
    assert apply_customer_category_overrides("mopro", product_name="Kondensmilch Dose") == "mopro"


def test_customer_category_api_only_normalizes_existing_category():
    assert apply_customer_category_overrides(" FISCH ") == "fisch"
    assert apply_customer_category_overrides("Obst & Gemüse") == "obst_gemuese"
    assert apply_customer_category_overrides("not-a-category") == "sonstiges"

    row = apply_customer_category_overrides_to_row(
        {"category": " FISCH ", "product_name": "Friesenkrone Heringssalat"}
    )
    assert row["category"] == "fisch"

    product = RawProduct(
        supplier="selgros",
        product_name="Kondensmilch Dose",
        category="mopro",
        unit="dose",
        price=1.49,
    )
    assert apply_customer_category_overrides_to_product(product) is product


def test_identity_prompt_mentions_customer_feedback_negative_examples():
    prompt = build_identity_prompt({"category": "obst_gemuese", "product_name": "Guacamole"})

    assert "Delikatess-Sahne-Heringsfilets => fisch, sauced, explicit_exclusion" in prompt
    assert "Pizzasauce => obst_gemuese, sauced, explicit_exclusion" in prompt
    assert "Guacamole => obst_gemuese, ready_to_eat, explicit_exclusion" in prompt
    assert "Gewürzgurken => obst_gemuese, pickled, explicit_exclusion" in prompt
    assert "Toilettenpapier => sonstiges, explicit_exclusion" in prompt
    assert "Frikadellen gebraten or gekochte Garnelen => cooked, explicit_exclusion" in prompt
    assert "rohe Cevapcici => fleisch, raw_formed, none" in prompt
    assert "Short Ribs BBQ => fleisch, processing unknown unless more evidence, none" in prompt
    assert "Avocado ready to eat => obst_gemuese, raw_plain, none" in prompt
    assert "Schweinelachse => fleisch, raw_cut, none; it is pork, not fish" in prompt
