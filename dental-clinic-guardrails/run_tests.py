# ============================================================
#  Batch test runner for the guarded dental agent
#  Reads test_prompts.csv (prompt, expected) and runs every row
#  through the real graph, then reports PASS/FAIL for each one.
# ============================================================

import csv                                                # csv = read/write CSV files
from pathlib import Path                                  # Path = build file paths next to this script

# import the pieces we already built in the main agent file
from dental_guardrail_agent import run_dental_agent, SAFE_REFUSAL

TEST_FILE = Path(__file__).parent / "test_prompts.csv"    # the input file: prompt, expected
RESULTS_FILE = Path(__file__).parent / "test_results.csv" # where we'll save the full results


def load_test_cases() -> list:
    """Read test_prompts.csv and return a list of {"prompt": ..., "expected": ...} dicts."""
    cases = []                                             # collect every row here
    with open(TEST_FILE, newline="", encoding="utf-8") as f:  # open the CSV file
        reader = csv.DictReader(f)                         # DictReader turns each row into {"prompt": ..., "expected": ...}
        for row in reader:                                 # go through every row
            cases.append({
                "prompt": row["prompt"].strip(),           # the message to send
                "expected": row["expected"].strip().lower(),  # "allow" or "block", lowercased
            })
    return cases


def run_one_test(prompt: str, expected: str) -> dict:
    """Run one prompt through the real graph and check if the outcome matches expected."""
    actual_response = run_dental_agent(prompt)             # run the FULL guarded graph, exactly like the CLI does
    # if the response is our known refusal text, the guard blocked it -> actual = "block"
    # anything else means Mia actually answered -> actual = "allow"
    actual = "block" if actual_response.strip() == SAFE_REFUSAL.strip() else "allow"
    passed = (actual == expected)                          # did the outcome match what we expected?
    return {
        "prompt": prompt,
        "expected": expected,
        "actual": actual,
        "passed": passed,
        "response": actual_response,
    }


def main() -> None:
    """Run every test case, print a summary, and save full results to CSV."""
    cases = load_test_cases()                              # load all rows from test_prompts.csv
    results = []                                            # collect every test's result

    print(f"Running {len(cases)} test prompts...\n")
    for i, case in enumerate(cases, start=1):               # start=1 so numbering starts at 1, not 0
        result = run_one_test(case["prompt"], case["expected"])
        results.append(result)
        status = "PASS" if result["passed"] else "FAIL"     # human-readable status
        # print one line per test so you see progress as it runs (this can take a minute for 40+ prompts)
        print(f"[{i}/{len(cases)}] {status}  expected={result['expected']:<6} actual={result['actual']:<6}  {case['prompt'][:60]}")

    # --- summary ---
    passed_count = sum(1 for r in results if r["passed"])   # count how many passed
    failed_count = len(results) - passed_count               # everything else failed
    print(f"\n{'='*60}")
    print(f"RESULTS: {passed_count}/{len(results)} passed, {failed_count} failed")
    print(f"{'='*60}\n")

    # --- print full detail for FAILURES only (the ones you need to fix) ---
    if failed_count > 0:
        print("FAILED TESTS (need attention):\n")
        for r in results:
            if not r["passed"]:
                print(f"  Prompt:   {r['prompt']}")
                print(f"  Expected: {r['expected']}   Actual: {r['actual']}")
                print(f"  Response: {r['response'][:150]}")
                print()

    # --- save everything to a CSV so you can review/share it later ---
    with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt", "expected", "actual", "passed", "response"])
        writer.writeheader()                                # write the column names as the first row
        writer.writerows(results)                           # write every result row
    print(f"Full results saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
