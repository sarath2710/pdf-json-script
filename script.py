import os
import re
import json
import requests
import tempfile
from pdf2image import convert_from_path
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ValidationError
from typing import List, Optional

# Load env variables
load_dotenv()
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_KEY)


#  Models of Pydantic---------------------------------------------------------------------------------------------------

class DocumentHeader(BaseModel):
    authority: Optional[str]
    office: Optional[str]
    page: Optional[str]


class CustomsDeclaration(BaseModel):
    net_weight: Optional[str]
    consignee_code: Optional[str]
    gross_weight: Optional[str]
    intercessor_co: Optional[str]
    measurement: Optional[str]
    commercial_reg_no: Optional[str]
    no_of_packages: Optional[str]
    export_to: Optional[str]
    marks_numbers: Optional[str]
    port_of_loading: Optional[str]
    port_of_discharge: Optional[str]
    destination: Optional[str]
    carrier_name: Optional[str]
    voyage_flight_no: Optional[str]
    bl_awb_manifest: Optional[str]


class GoodsDetails(BaseModel):
    loc: Optional[str]
    total_duty: Optional[str]
    hs_code: Optional[str]
    goods_description: Optional[str]
    origin: Optional[str]
    foreign_value: Optional[str]
    currency_type: Optional[str]
    exchange_rate: Optional[str]
    cif_local_value: Optional[str]
    duty_rate: Optional[str]
    income_type: Optional[str]
    total_duty_type: Optional[str]


class AdditionalGoods(BaseModel):
    hs_code: Optional[str]
    description: Optional[str]
    origin: Optional[str]
    foreign_value: Optional[str]
    currency: Optional[str]
    exchange_rate: Optional[str]
    cif_local_value: Optional[str]
    duty_rate: Optional[str]
    duty_type: Optional[str]
    total_duty: Optional[str]


class ExemptionDuty(BaseModel):
    qty: Optional[str]
    type: Optional[str]
    qty_2: Optional[str]
    unit: Optional[str]
    net: Optional[str]
    gross: Optional[str]
    clearing_agent: Optional[str]
    license_no: Optional[str]
    agency: Optional[str]
    release_ref: Optional[str]
    sources: Optional[str]
    code: Optional[str]
    beneficiary: Optional[str]


class DutiesAndFees(BaseModel):
    inspection: Optional[str]
    inspector: Optional[str]
    group_supervisor: Optional[str]
    other_remarks: Optional[str]
    release_date: Optional[str]
    route: Optional[str]
    exit_port: Optional[str]
    exit_transaction_no: Optional[str]
    date: Optional[str]
    security_officer: Optional[str]
    transit_officer: Optional[str]
    total_duty: Optional[str]
    handling: Optional[str]
    storage: Optional[str]
    other_charges: Optional[str]
    definite: Optional[str]
    total_fee: Optional[str]
    payment_method: Optional[str]
    guarantee_cheque: Optional[str]
    duty: Optional[str]
    date_2: Optional[str]
    bank: Optional[str]
    receipt_no: Optional[str]
    bank_2: Optional[str]


class FooterNotes(BaseModel):
    distribution: Optional[str]


class Invoice(BaseModel):
    document_header: Optional[DocumentHeader]
    port_type: Optional[str]
    dec_type: Optional[str]
    dec_date: Optional[str]
    dec_no: Optional[str]
    customs_declaration: Optional[CustomsDeclaration]
    goods_details: Optional[GoodsDetails]
    additional_goods: Optional[List[AdditionalGoods]]
    exemption_duty: Optional[ExemptionDuty]
    duties_and_fees: Optional[DutiesAndFees]
    footer_notes: Optional[FooterNotes]


class Declaration(BaseModel):
    document_header: Optional[DocumentHeader]
    port_type: Optional[str]
    dec_type: Optional[str]
    dec_date: Optional[str]
    dec_no: Optional[str]
    customs_declaration: Optional[CustomsDeclaration]
    goods_details: Optional[GoodsDetails]
    additional_goods: Optional[List[AdditionalGoods]]
    exemption_duty: Optional[ExemptionDuty]
    duties_and_fees: Optional[DutiesAndFees]
    footer_notes: Optional[FooterNotes]


# PDF to text-------------------------------------------------------------------------------------------------

def image_to_text_pil(pil_img):
    """Sending PIL images into the OCR API and return extracted text."""
    url = "https://ocr-extract-text.p.rapidapi.com/ocr"

    headers = {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": "ocr-extract-text.p.rapidapi.com"
    }

    with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
        pil_img.save(tmp, format="JPEG")
        tmp.seek(0)
        files = {"image": tmp}
        response = requests.post(url, headers=headers, files=files)

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error {response.status_code}: {response.text}")


def pdf_to_text(pdf_path):
    """Convert a PDF into text (page by page)"""
    images = convert_from_path(pdf_path)
    results = []

    for i, img in enumerate(images, start=1):
        try:
            text_result = image_to_text_pil(img)
            results.append({"page": i, "text": text_result})
        except Exception as e:
            results.append({"page": i, "error": str(e)})

    return results

# Prompts------------------------------------------------------------------------------------------------

prompt_detect = """You are a classifier. Determine whether the following text is an "Invoice" or a "Declaration". 
Respond with only one word: Invoice or Declaration. 

Text:
{input_text}
"""

prompt_declaration = """You are an extraction assistant.
Task: Convert the following declaration into structured JSON.

Example:
Input: "Declaration by United Arab Emirates Federal Customs Authority Dubai Customs"
Output: "{
  "document_header": {
    "authority": "UNITED ARAB EMIRATES Federal Customs Authority",
    "office": "DUBAI CUSTOMS",
    "page": "Page 1 of 1"
  },
  "port_type": "SEA",
  "dec_type": "IMPORT",
  "dec_date": "17/09/2024",
  "dec_no": "101-25367688-24",
  "customs_declaration": {
    "net_weight": "318 kg",
    "consignee_code": "41111514795",
    "gross_weight": "318 kg",
    "intercessor_co": "AE-2036625 - NOURGES GENERAL TRADING FZE (I - N8817)",
    "measurement": "",
    "commercial_reg_no": "AE-1002540 - RELIANCE FREIGHT SYSTEMS",
    "no_of_packages": "1 - PALLET",
    "export_to": "420344201",
    "marks_numbers": "GAOU6198888 SEAL NO. 215435 N/ M",
    "port_of_loading": "BUSAN",
    "port_of_discharge": "JEBEL ALI",
    "destination": "",
    "carrier_name": "HMM MIR",
    "voyage_flight_no": "924009",
    "bl_awb_manifest": "FBDXB240354"
  },
  "goods_details": {
    "loc": "JEBEL ALI PR-00266/RELIANCE FREIGHT SYSTEMS L",
    "total_duty": "AED 690.44",
    "hs_code": "33049990",
    "goods_description": "MADAGASCAR CENTELLA TO",
    "origin": "KR",
    "foreign_value": "3680.00",
    "currency_type": "USD",
    "exchange_rate": "3.6930",
    "cif_local_value": "13608.74",
    "duty_rate": "5.0",
    "income_type": "DEF",
    "total_duty_type": "AED"
  },
  "additional_goods": [
    {
      "hs_code": "33049990",
      "description": "MADAGASCAR CENTELLA HY",
      "origin": "KR",
      "foreign_value": "325.00",
      "currency": "USD",
      "exchange_rate": "3.6930",
      "cif_local_value": "1219.52",
      "duty_rate": "5.0",
      "duty_type": "DEF",
      "total_duty": "60.98"
    },
    {
      "hs_code": "33049990",
      "description": "MADAGASCAR CENTELLA AM",
      "origin": "KR",
      "foreign_value": "4205.00",
      "currency": "USD",
      "exchange_rate": "3.6930",
      "cif_local_value": "15778.74",
      "duty_rate": "5.0",
      "duty_type": "DEF",
      "total_duty": "788.94"
    }
  ],
  "exemption_duty": {
    "qty": "1",
    "type": "PLT",
    "qty_2": "",
    "unit": "kg",
    "net": "318 kg",
    "gross": "318",
    "clearing_agent": "AE-1002540 RELIANCE FREIGHT SYSTEMS",
    "license_no": "246751",
    "agency": "",
    "release_ref": "CPRE-2024-10 BMH HEALTH&SFTY",
    "sources": "",
    "code": "",
    "beneficiary": ""
  },
  "duties_and_fees": {
    "inspection": "Container Nos: GAOU6198888",
    "inspector": "",
    "group_supervisor": "",
    "other_remarks": "[FOB] FRT: 184.65 INS: 302.83 Total Value: 30319.53 Controlling Authority Inspection required",
    "release_date": "17/09/2024",
    "route": "",
    "exit_port": "",
    "exit_transaction_no": "",
    "date": "",
    "security_officer": "",
    "transit_officer": "",
    "total_duty": "1540",
    "handling": "",
    "storage": "90",
    "other_charges": "",
    "definite": "1630",
    "total_fee": "",
    "payment_method": "",
    "guarantee_cheque": "",
    "duty": "1540 [10472602] CA-1002089",
    "date_2": "",
    "bank": "",
    "receipt_no": "",
    "bank_2": ""
  },
  "footer_notes": {
    "distribution": "1 - Handling Authority 2 - Consignee 3 - Consignee 4 - Customs"
  }
}
"

Now process:
{input_text}
"""

prompt_invoice = """You are an extraction assistant.
Task: Convert the following invoice into structured JSON.

Example:
Input: "Invoice for Craver corporation"
Output: "{
  "document_consumer_products_registration_certificate": {
    "header": {
      "authority": "GOVERNMENT OF DUBAI",
      "municipality": "DUBAI MUNICIPALITY",
      "department": "Public Health & Safety Department",
      "section": "Consumer Products Safety Section",
      "document_type": "Consumer Products Registration Certificate",
      "product_category": "Cosmetics and Personal care Products"
    },
    "reference_number": "CPRE-2024-117190",
    "registration_status": "Approved",
    "registration_date": "10/07/2024",
    "category": "Face & Neck Preparations",
    "brand_name": "SKIN1004",
    "company_name": "Nourges General Trading FZE",
    "country_of_origin": "South Korea",
    "product_details": {
      "product_name": "SKIN1004 HYALU - CICA WATER - FIT SUN SERUM SPF 50+"
    },
    "important_notes": {
      "validity": "Registration Certificate is valid for 5 years from date of issue & any alteration or deletion in any way will invalid it",
      "distribution": "Registration Certificate is granted upon company's request and the above listed product is freely distributed in the local market & Dubai Municipality will not be responsible for any consequences of variations in the product",
      "regulations": "Registration Certificate is issued upon the currently enforced regulations and subjected for modification according to the requirements of concerned department",
      "inspection": "All the imported & marketed products will be subjected to inspection and conformity with the currently enforced regulations",
      "electronic_certificate": "This certificate has been electronically generated and approved by Municipality and does not need signature or Stamp"
    },
    "footer": {
      "section_name": "Consumer Products Safety Section",
      "vision": "Creating an excellent city that provides the essence of success and comfort of sustainable living",
      "contact": {
        "phone_1": "+97142246565",
        "phone_2": "+97142215555",
        "address": "P.O.Box: 67 DUBAI, UAE",
        "tel": "+971 4 221 5555",
        "fax": "+971 4 224 6656",
        "email": "info@dm.gov.ae",
        "website": "www.dm.gov.ae"
      }
    }
  },
  "document_invoice": {
    "document_type": "INVOICE",
    "shipper": {
      "company": "CRAVER CORPORATION",
      "address": "403 13 dong BAEKJE MAUL MATJUMTOWER TEHERAN-RO 4-GIL GANGNAM-GU, SEOUL KOREA 06232",
      "contact": "ATTN: LOGISTICS / SARAH KIM TEL: +82 70 5080 4883"
    },
    "date": "2024.07.25",
    "invoice_number": "2024246712343016",
    "incoterms": "FOB",
    "country_of_origin": "REPUBLIC OF KOREA",
    "consignee": {
      "company": "Nourges General Trading FZE",
      "address": "Business Center, Sharjah, United Arab Emirates 00000",
      "contact": "attn: RIZWAN RAFIQ / rafiq@nourges.com, support@nourges.com, rizwan541514@gmail.com / +971556151945"
    },
    "notify_party": {
      "company": "Nourges General Trading FZE",
      "address": "Business Center, Sharjah, United Arab Emirates 00000",
      "contact": "attn: RIZWAN RAFIQ / rafiq@nourges.com, support@nourges.com, rizwan541514@gmail.com / +971556151945"
    },
    "product_details": [
      {
        "upc": "8809576408848",
        "sku": "BLJ-ATMFMSEEHO0700",
        "hs_code": "3304.99.3000",
        "description": "SKIN1004 Madagascar Centella Ampoule 100ml",
        "unit_price_usd": "6.43",
        "qty_ea": "900",
        "amount_usd": "4590.00"
      },
      {
        "upc": "8809576263172",
        "sku": "BLJ-ATMFMSEEHO1500",
        "hs_code": "3304.99.3000",
        "description": "SKIN1004 Madagascar Centella Tone Brightening Capsule Ampoule 100ml",
        "unit_price_usd": "7.10",
        "qty_ea": "600",
        "amount_usd": "3190.00"
      },
      {
        "upc": "8809576265152",
        "sku": "BLJ-SHFMSOGAMSP900",
        "hs_code": "3304.99.3000",
        "description": "SKIN1004 Madagascar Centella Hyalu-Cica Water Fit Sun Serum 50ml",
        "unit_price_usd": "6.05",
        "qty_ea": "50",
        "amount_usd": "325.00"
      }
    ],
    "totals": {
      "ttl_amount_usd": "8210.00",
      "ttl_qty_ea": "1550",
      "ttl_net_weight_kg": "311.30",
      "ttl_gross_weight_kg": "317.65"
    },
    "signature_section": {
      "reference_number": "261-81-1484",
      "company": "CRAVER CORPORATION",
      "address": "Mapo-gu 12F 14 Teheran-ro Gangnam-gu, Seoul, Korea",
      "business_type": "Wholesale and retail trade"
    }
  },
  "document_variants_information": {
    "header": {
      "authority": "GOVERNMENT OF DUBAI",
      "municipality": "DUBAI MUNICIPALITY"
    },
    "document_type": "Variants Information",
    "product_info": {
      "product_name": "SKIN1004 Hyalu-CICA Water-fit Sun Serum 50ml",
      "international_barcode": "8809576261592",
      "scent_flavor": "Normal",
      "product_color_shade": "PALE YELLOW",
      "size_weight_volume": "50",
      "unit": "mL"
    },
    "footer": {
      "vision": "Creating an excellent city that provides the essence of success and comfort of sustainable living",
      "contact": {
        "phone_1": "+97142246565",
        "phone_2": "+97142215555",
        "address": "P.O.Box: 67 DUBAI, UAE",
        "tel": "+971 4 221 5555",
        "fax": "+971 4 224 6656",
        "email": "info@dm.gov.ae",
        "website": "www.dm.gov.ae"
      }
    }
  }
}
"

Now process:
{input_text}
"""

# LLM Functions -----------------------------------------------------------------------------------------------------

def load_prompt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def classify_text(input_text: str) -> str:
    """Using LLM (OpenAI) for detect type (Declaration / Invoice)"""
    # prompt = load_prompt("prompt_detect.txt").replace("{input_text}", input_text) 
    filled_prompt = prompt_detect.replace("{input_text}", input_text)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a document classifier. Reply with only one word: Invoice or Declaration."},
            {"role": "user", "content": filled_prompt}
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip()


def clean_json(raw_text: str) -> str:
    """Cleaning json"""
    match = re.search(r"```json\s*(.*?)```", raw_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```(.*?)```", raw_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return raw_text.strip()


def extract_structured(input_text: str, prompt_template: str = None) -> dict:
    """Send text to LLM + validate with Pydantic."""
    doc_type = classify_text(input_text)

    if doc_type.lower() == "declaration":
        # prompt = (prompt_template or load_prompt("prompt_declaration.txt")).replace("{input_text}", input_text)
        filled_prompt = prompt_declaration.replace("{input_text}", input_text)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Return ONLY valid JSON. Do not add explanations or code fences."},
                {"role": "user", "content": filled_prompt}
            ],
            temperature=0
        )
        raw_json = clean_json(response.choices[0].message.content.strip())
        try:
            data = json.loads(raw_json)
            return Declaration(**data).model_dump()
        except (json.JSONDecodeError, ValidationError) as e:
            raise ValueError(f"Failed to validate Declaration: {e}\nRaw output:\n{raw_json}")

    elif doc_type.lower() == "invoice":
        # prompt = (prompt_template or load_prompt("prompt_invoice.txt")).replace("{input_text}", input_text)
        filled_prompt = prompt_invoice.replace("{input_text}", input_text)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Return ONLY valid JSON. Do not add explanations or code fences."},
                {"role": "user", "content": filled_prompt}
            ],
            temperature=0
        )
        raw_json = clean_json(response.choices[0].message.content.strip())
        try:
            data = json.loads(raw_json)
            return Invoice(**data).model_dump()
        except (json.JSONDecodeError, ValidationError) as e:
            raise ValueError(f"Failed to validate Invoice: {e}\nRaw output:\n{raw_json}")

    else:
        raise ValueError(f"Unknown document type: {doc_type}")


# Main usage----------------------------------------------------------------------------------------------

if __name__ == "__main__":
    pdf_path = "D:\SynkCode\pdf-json-demo\sample image.pdf"  
    extracted_texts = pdf_to_text(pdf_path)

    all_text = " ".join([p["text"]["text"] for p in extracted_texts if "text" in p and "text" in p["text"]])
    print("DEBUG: Extracted text length =", len(all_text))

    structured = extract_structured(all_text)
    print(json.dumps(structured, indent=2))
