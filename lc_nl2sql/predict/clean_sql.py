import sys
import re

def clean_file(input_path, output_path):
    with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
        for line in f_in:
            line = line.strip()
            # 1. Search for pattern SELECT ... ;
            match = re.search(r'(SELECT\s.*?;)', line, re.IGNORECASE)
            
            if match:
                sql = match.group(1)
            else:
                # 2. If there is no semicolon, try to take from SELECT to the end
                # or clean markdown characters ```sql ... ```
                clean_line = line.replace("```sql", "").replace("```", "")
                idx = clean_line.upper().find("SELECT")
                if idx != -1:
                    sql = clean_line[idx:]
                else:
                    # If there is no SELECT at all (model went off track), provide a dummy SQL to avoid crashing
                    sql = "SELECT 1" 
            
            f_out.write(sql + "\n")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python clean_sql.py <input_file> <output_file>")
        sys.exit(1)
    
    clean_file(sys.argv[1], sys.argv[2])
    print(f"Cleaned SQLs saved to: {sys.argv[2]}")