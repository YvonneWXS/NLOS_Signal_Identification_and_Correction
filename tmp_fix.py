path = r'D:\3_document\4_research\NLOS Signal Identification and Correction\model\part1_GAT\model\run_serial.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Fix SRC_DIR
old1 = 'RadioGAT-Multi-band-Radiomap-Reconstruction'
new1 = 'model'
content = content.replace(old1, new1)

# Fix experiments
old2 = 'exp_034'
new2 = 'exp_052'
content = content.replace(old2, new2)
content = content.replace('exp_035', 'exp_053')
content = content.replace('exp_036', 'exp_054')
content = content.replace('exp_037', 'exp_055')

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print('run_serial.py updated')
