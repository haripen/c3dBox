import pandas as pd
from io import StringIO
def load_osimFile(file_path, cols2return):
    """
    Tested for IK, IK marker errors, ID, SO force, SO activation, and JRL
    """
    if isinstance(cols2return, str):
        cols2return = (cols2return,)
    with open(file_path, 'r') as file:
        while True:
            line = file.readline().strip()
            if line == 'endheader':
                break
        header_line = file.readline().replace('\t', ' ').strip()
        header_line = ' '.join(header_line.split())
        data_lines = file.readlines()
    cleaned_data = [' '.join(line.split()) for line in data_lines]
    cleaned_file_content = header_line + '\n' + '\n'.join(cleaned_data)
    df = pd.read_csv(StringIO(cleaned_file_content), sep=' ')
    cols2return = ('time',) + cols2return if 'time' not in cols2return else cols2return
    df = df[list(cols2return)]
    df.set_index('time', inplace=True)
    return df