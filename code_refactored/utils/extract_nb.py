import json

path = r"c:\Users\lehoa\dự_án_NTS\code_refactored\project_notebook.ipynb"
with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)

in_step_3 = False
for cell in nb.get("cells", []):
    if cell["cell_type"] == "markdown":
        src = "".join(cell.get("source", []))
        if "Bước 3" in src:
            in_step_3 = True
            print("--- START STEP 3 ---")
        elif "Bước 4" in src:
            in_step_3 = False
            print("--- END STEP 3 ---")
    if in_step_3 and cell["cell_type"] == "code":
        src = "".join(cell.get("source", []))
        print("CODE CELL:\n" + src + "\n" + "="*40)
