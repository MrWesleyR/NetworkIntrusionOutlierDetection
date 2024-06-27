import arff

def find_label_index(attributes, label_name):
    for i, (attr_name, attr_values) in enumerate(attributes):
        if attr_name == label_name and set(attr_values) == {'BENIGN', 'DDoS'}:
            return i
    raise ValueError(f"Label attribute '{label_name}' not found or has invalid values.")

def dividir_arquivo_arff(arquivo_entrada, arquivo_saida_benigno, arquivo_saida_ddos):
    with open(arquivo_entrada, 'r') as f:
        dataset = arff.load(f)

    meta = dataset['attributes']
    data = dataset['data']

    label_index = find_label_index(meta, ' Label')

    exemplos_benignos = [exemplo for exemplo in data if exemplo[label_index] == 'BENIGN']
    exemplos_ddos = [exemplo for exemplo in data if exemplo[label_index] == 'DDoS']

    with open(arquivo_saida_benigno, 'w') as f:
        arff.dump({'attributes': meta, 'data': exemplos_benignos, 'relation': 'benignos'}, f)

    with open(arquivo_saida_ddos, 'w') as f:
        arff.dump({'attributes': meta, 'data': exemplos_ddos, 'relation': 'ddos'}, f)

# Exemplo de uso
arquivo_entrada = 'data/Friday-CICIDS.arff'
arquivo_saida_benigno = 'benignos.arff'
arquivo_saida_ddos = 'ddos.arff'

dividir_arquivo_arff(arquivo_entrada, arquivo_saida_benigno, arquivo_saida_ddos)
