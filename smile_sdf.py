import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, SDWriter
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)

# 读取 CSV
df = pd.read_csv('data/EX2_label2.csv')
writer = SDWriter('data/EX2_label2.sdf')

for idx, row in df.iterrows():
    smiles = row['SMILES']
    label = row['Label2']
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logging.warning(f"无法解析 SMILES: {smiles}")
        continue
    
    # 添加氢（可选）
    mol = Chem.AddHs(mol)
    
    # 尝试嵌入 3D 坐标
    try:
        # 使用随机坐标，固定随机种子以保证可重复性
        AllChem.EmbedMolecule(mol, randomSeed=42)
        # 检查是否成功生成构象
        if mol.GetNumConformers() == 0:
            logging.warning(f"无法为 {smiles} 生成构象")
            continue
        # 尝试力场优化
        if AllChem.MMFFHasAllMoleculeParams(mol):
            AllChem.MMFFOptimizeMolecule(mol)
        else:
            # 若 MMFF 不可用，使用 UFF
            AllChem.UFFOptimizeMolecule(mol)
    except Exception as e:
        logging.error(f"生成构象失败 {smiles}: {e}")
        continue
    
    # 设置属性
    mol.SetProp('Label2', str(label))
    writer.write(mol)

writer.close()
logging.info("SDF 文件生成完成")