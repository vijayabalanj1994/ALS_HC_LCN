import pandas as pd
from config import config

LABELS = ["Concordant", "Discordant", "Control"]

def make_five_folds_by_index(df):
    # df is expected to be cleaned/normalized already

    # Ordered list of Case IDs per label (order of first appearance)
    cases_by_label = {
        lab: df.loc[df["Category"] == lab, "Case ID"].drop_duplicates().tolist()
        for lab in LABELS
    }

    # Require exactly 5 cases per category
    for lab in LABELS:
        n = len(cases_by_label[lab])
        if n != 5:
            raise ValueError(f"{lab}: expected 5 cases, found {n}: {cases_by_label[lab]}")

    all_cases = set(sum(cases_by_label.values(), []))  # union of all cases across labels

    folds = []
    for i in range(5):
        # pick index i from each category for the test set of fold i
        test_cases = {lab: cases_by_label[lab][i] for lab in LABELS}
        test_case_set = set(test_cases.values())
        train_case_set = all_cases - test_case_set

        test_df  = df[df["Case ID"].isin(test_case_set)].reset_index(drop=True)
        train_df = df[df["Case ID"].isin(train_case_set)].reset_index(drop=True)

        folds.append({
            "fold": i,
            "test_cases": test_cases,   # one per category
            "train_df": train_df,
            "test_df": test_df,
        })

    return folds

def get_folds():
    file_path = config.keys_path
    df = pd.read_excel(file_path, header=1)

    # Normalize Case IDs
    df["Case ID"] = df["Case ID"].astype(str).str.split().str[0]
    df["Case ID"] = df["Case ID"].replace("SD12-17", "SD012-17")  # fix typo

    # ---- Overall summary (Case ID, Category, count) ----
    overall = (
        df.groupby(["Case ID", "Category"]).size()
          .reset_index(name="count")
          .sort_values(["Category", "Case ID"])
    )
    print("\nOverall summary (Case ID, Category, count):")
    print(overall.to_string(index=False))

    # ---- Build 5 folds by index rule ----
    folds = make_five_folds_by_index(df)

    # ---- Report each fold ----
    for f in folds:
        i = f["fold"]
        test_cases = f["test_cases"]
        print(f"\nFold {i} \ntest cases:")
        for lab in LABELS:
            print(f"  {lab}: {test_cases[lab]}")

        train_counts = f["train_df"]["Category"].value_counts().to_dict()
        test_counts  = f["test_df"]["Category"].value_counts().to_dict()
        print("Train label distribution:\n", train_counts)
        print("Test  label distribution:\n",  test_counts)

        # sizes and percentages
        train_n = len(f["train_df"])
        test_n  = len(f["test_df"])
        total   = train_n + test_n
        print(f"Train size: {train_n}/{total} ({train_n/total:.1%})")
        print(f"Test  size: {test_n}/{total} ({test_n/total:.1%})")

    return folds

if __name__ == "__main__":
    folds = get_folds()
