---
return "", 0
# Use partial token set ratio for resilience to rephrasing
scores = [fuzz.token_set_ratio(requirement, ch) for ch in submittal_chunks]
idx = int(np.argmax(scores))
return submittal_chunks[idx], int(scores[idx])


# ------------------------------- UI -----------------------------------------
left, right = st.columns(2, gap="large")


with left:
st.subheader("Spec Section")
spec_text = st.text_area(
"Paste the relevant spec section:", height=320,
value=(
"PART 1 – GENERAL\n" \
"1.02 SUBMITTALS\n" \
"A. Product Data: Provide manufacturer's data sheets.\n" \
"B. Shop Drawings: Submit coordinated drawings.\n" \
"C. Certificates: Provide compliance certifications.\n" \
"D. Warranty: Minimum one year warranty.\n"
)
)


with right:
st.subheader("Submittal Text")
submittal_text = st.text_area(
"Paste the contractor's submittal narrative:", height=320,
value=(
"We are submitting product data for review.\n" \
"Included: manufacturer data sheets and a warranty statement.\n"
)
)


threshold = st.slider("Match threshold (0‑100)", min_value=50, max_value=100, value=78, help="Lower = more forgiving matches")
run = st.button("Analyze")


if run:
spec_lines = split_lines(spec_text)
reqs = slice_requirement_candidates(spec_lines)


# Submittal chunks: use lines and paragraph windows for better recall
sub_lines = split_lines(submittal_text)
sub_chunks = sub_lines[:]
# Also add 2‑line windows
for i in range(len(sub_lines) - 1):
sub_chunks.append(sub_lines[i] + " " + sub_lines[i + 1])


results: List[MatchResult] = []
for r in reqs:
m, s = best_match(r, sub_chunks)
status = "Found" if s >= threshold else ("Weak" if s >= max(60, threshold - 10) else "Missing")
results.append(MatchResult(r, m, s, status))


df = pd.DataFrame([{
"Requirement": r.requirement,
"Best Match In Submittal": r.matched_text,
"Score": r.score,
"Status": r.status,
} for r in results])


found = (df["Status"] == "Found").sum()
weak = (df["Status"] == "Weak").sum()
missing = (df["Status"] == "Missing").sum()
coverage = int(round(100 * found / max(1, len(df))))


st.success(f"Coverage: {coverage}% | Found: {found} • Weak: {weak} • Missing: {missing}")


st.dataframe(df, use_container_width=True, hide_index=True)


st.download_button(
label="Download results as CSV",
data=df.to_csv(index=False).encode("utf-8"),
file_name="submittal_check_results.csv",
mime="text/csv",
)


st.info("Heuristic tool only. Always read the spec and submittal—this is a triage helper, not a final judgment.")
else:
st.markdown("Click **Analyze** to run the comparison. Try adjusting the threshold if matches feel too strict/loose.")
