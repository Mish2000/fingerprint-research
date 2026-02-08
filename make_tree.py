import os

# הגדרות
root_dir = r"C:\fingerprint-research"
output_file = r"C:\fingerprint-research\project_tree.txt"
ignore_dirs = {'node_modules', '.git', '.idea', '__pycache__'}


def generate_tree(startpath, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for root, dirs, files in os.walk(startpath):
            # הסרת תיקיות שאנחנו לא רוצים לסרוק (משנה את הרשימה במקום)
            dirs[:] = [d for d in dirs if d not in ignore_dirs]

            # חישוב העומק לצורך הזחה (אינדנטציה)
            level = root.replace(startpath, '').count(os.sep)
            indent = '    ' * level

            # כתיבת שם התיקייה
            f.write(f'{indent}[{os.path.basename(root)}]\n')

            # כתיבת הקבצים בתיקייה
            subindent = '    ' * (level + 1)
            for file in files:
                f.write(f'{subindent}{file}\n')


if __name__ == "__main__":
    generate_tree(root_dir, output_file)
    print(f"Tree generated at: {output_file}")