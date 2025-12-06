"""
Quick test script to verify setup
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("="*60)
print("Testing Math Benchmark Setup")
print("="*60)

# Test 1: Check API Key
print("\n1. Checking OpenAI API Key...")
api_key = os.getenv('OPENAI_API_KEY')
if api_key and api_key.startswith('sk-'):
    print("   [OK] API Key found and valid format")
else:
    print("   [FAIL] API Key not found or invalid")
    exit(1)

# Test 2: Test OpenAI API connection
print("\n2. Testing OpenAI API connection...")
try:
    import openai
    client = openai.OpenAI(api_key=api_key)

    # Simple test call
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "What is 2+2?"}],
        max_tokens=10
    )

    answer = response.choices[0].message.content
    print(f"   [OK] API connected successfully!")
    print(f"   Response: {answer}")

except Exception as e:
    print(f"   [FAIL] API connection failed: {e}")
    exit(1)

# Test 3: Check required packages
print("\n3. Checking required packages...")
required_packages = ['torch', 'transformers', 'datasets', 'yaml', 'pandas', 'matplotlib']
for pkg in required_packages:
    try:
        __import__(pkg)
        print(f"   [OK] {pkg}")
    except ImportError:
        print(f"   [FAIL] {pkg} not installed")

# Test 4: Load a small dataset
print("\n4. Testing dataset loading...")
try:
    from datasets import load_dataset
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    print(f"   [OK] GSM8K dataset loaded: {len(dataset)} examples")

    # Show a sample
    sample = dataset[0]
    print(f"\n   Sample question: {sample['question'][:100]}...")

except Exception as e:
    print(f"   [FAIL] Dataset loading failed: {e}")

print("\n" + "="*60)
print("Setup Test Complete!")
print("="*60)
print("\nYou can now run the benchmark:")
print("  python run_benchmark.py --models gpt-4o-mini --datasets gsm8k")
