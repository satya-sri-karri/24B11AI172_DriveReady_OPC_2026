# Day 4 – AutoMate AI with Google Sheets

## Aim
Extend the AutoMate AI automotive assistant with tools that read dealership inventory, estimate loan instalments and store enquiries in a Google Sheet.

## What this task covers
1. Authenticating Google Sheets in Google Colab using `gspread`.
2. Creating or reusing a workbook with `Inventory` and `Leads` tabs.
3. Defining LangChain tools for live inventory lookup, EMI estimation and lead capture.
4. Running a tool-using agent with Groq's LLM.
5. Testing the assistant with the scenarios in `test.csv`.

## Sheet layout
**Inventory:** `vehicle_id,make,model,variant,year,asking_price_inr,fuel_type,transmission,odometer_km,location,status,last_verified_at`

**Leads:** `lead_id,created_at_utc,customer_name,contact,budget_max_inr,preferred_vehicle,location,notes,consent_confirmed`

Only inventory records with status `AVAILABLE` are included in search results. Populate the Inventory tab with authorised example or dealership data. The program does not generate fake stock records.

## Run in Google Colab
1. Open a notebook and install packages:
   `!pip -q install -U langchain langchain-groq gspread google-auth`
2. Upload `task.py` and run it with `%run task.py`.
3. Add a `GROQ_API_KEY` in **Colab → Secrets**, enable notebook access and approve Google authentication.
4. Open the workbook link printed by the program; enter some vehicle rows in Inventory.
5. Ask inventory, EMI and lead-management questions. Use `reset` to forget conversation history or `exit` to stop.

## Files
- `task.py`: Google Sheets connection, three tools, and the interactive AutoMate AI agent
- `test.csv`: natural-language input examples and the expected behaviour

## Precautions and limitations
- The agent writes personal contact information only when consent is explicit; test with non-sensitive sample data.
- A loan EMI is an **estimate**, not a lender quote; actual interest, processing fees and terms may vary.
- Google account authorisation and external APIs are necessary to execute this project.
- No messaging, payment, booking or ownership-verification tools are connected.
- Live API execution has not been verified here; run each test case in your own authorised Colab session.
