import scipy.io
# Load .mat file
mat_data = scipy.io.loadmat('case_1.mat')
# List all variables
print("Variables in .mat file:")
for key in mat_data.keys():
    if "__" not in key:  # This will exclude meta entries
        print(key)
# Access variables
x0_nlp = mat_data['x0_nlp']
print(x0_nlp)
# variable names: 
# x0_nlp
# lamx0_nlp
# lamg0_nlp