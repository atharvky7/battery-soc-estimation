import pandas as pd
import scipy.io as sio
import os
import json

def analyze_excel(filepath):
    try:
        print(f"Analyzing {filepath}...")
        # Read a small chunk first to see columns
        df_head = pd.read_excel(filepath, nrows=5)
        columns = df_head.columns.tolist()
        
        # We might not want to read the entire 60MB file if it takes too long, but let's try reading it completely for stats.
        df = pd.read_excel(filepath)
        
        stats = {
            'rows': len(df),
            'columns': columns,
            'summary': {}
        }
        
        for col in columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                stats['summary'][col] = {
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'mean': float(df[col].mean()),
                    'nulls': int(df[col].isnull().sum())
                }
            else:
                stats['summary'][col] = {
                    'unique_values': int(df[col].nunique()),
                    'nulls': int(df[col].isnull().sum())
                }
        
        return stats
    except Exception as e:
        return {'error': str(e)}

def analyze_mat(filepath):
    try:
        print(f"Analyzing {filepath}...")
        mat = sio.loadmat(filepath)
        keys = [k for k in mat.keys() if not k.startswith('__')]
        return {
            'keys': keys,
            'file_size_mb': os.path.getsize(filepath) / (1024*1024)
        }
    except Exception as e:
        return {'error': str(e)}

if __name__ == "__main__":
    results = {}
    
    excel_path = r"C:\Users\mahes\Downloads\Major Project\EV Lithium Ion Battery State of Charge Dataset\EV Lithium Ion Battery State of Charge Dataset\charging and discharging (1).xlsx"
    if os.path.exists(excel_path):
        results['Mendeley_Dataset'] = analyze_excel(excel_path)
    
    mat_path = r"C:\Users\mahes\Downloads\Major Project\Oxford_Battery_Degradation_Dataset_1.mat"
    if os.path.exists(mat_path):
        results['Oxford_Dataset'] = analyze_mat(mat_path)
        
    print(json.dumps(results, indent=2))
