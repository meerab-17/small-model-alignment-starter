import os  
import zipfile 

def make_deliverables_zip(zip_path, max_file_mb=50):
    items = ["scripts", "eval_sets", "data", "tests", "outputs",
             "submission_adapter.py", "requirements.txt", "pyproject.toml"]
    skipped = []  
    added = 0  

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in items:
            if os.path.isfile(item):  
                zf.write(item)
                added = added + 1
            elif os.path.isdir(item): 
                for root, dirs, files in os.walk(item):
                    for name in files:
                        full_path = os.path.join(root, name)
                        size_mb = os.path.getsize(full_path) / (1024 * 1024)
                        if size_mb > max_file_mb:  # too big for github, leave it out
                            skipped.append(full_path)
                        else:
                            zf.write(full_path)
                            added = added + 1
    print("files in zip:", added, "| skipped because over", max_file_mb, "MB:", len(skipped))
    for path in skipped:
        print("  skipped:", path)
