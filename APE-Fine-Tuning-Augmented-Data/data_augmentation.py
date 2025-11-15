import pandas as pd
from openai import OpenAI
import time

#  Connect to OpenRouter 
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="API_KEY",
)

#  Load dataset 
# !git clone https://github.com/dml-qom/FarsTail.git
# !ls FarsTail/data

test = pd.read_csv('FarsTail/data/Test-word.csv', sep='\t')
train = pd.read_csv('FarsTail/data/Train-word.csv', sep='\t')
val = pd.read_csv('FarsTail/data/Val-word.csv', sep='\t')

df = train.iloc[0:100]  

#  Function to generate reason 
def generate_reason(premise, hypothesis, label):
    prompt = f"""
Premise: "{premise}"
Hypothesis: "{hypothesis}"
Label: {label}

Please write a short and natural explanation in Persian explaining why this label is correct.
"""
    try:
        response = client.chat.completions.create(
            model="google/gemma-3-27b-it:free",  
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        reason = response.choices[0].message.content.strip()
        return reason
    except Exception as e:
        print("Error:", e)
        return ""

#  Add reason column 
reasons = []
for idx, row in df.iterrows():
    print(f"Processing row {idx+1}/{len(df)} ...")
    reason = generate_reason(row["premise"], row["hypothesis"], row["label"])
    reasons.append(reason)
    time.sleep(1)  # Short delay to avoid API rate limiting

df["reason"] = reasons

#  Save the new dataset 
df.to_csv("textual_entailment_with_reason_1.csv", index=False, encoding="utf-8-sig")

print(" Reason column added and file saved successfully.")