



with open("workload_azure.txt", "r") as f:
    txt = f.read()
    
workload = list(filter(lambda x: x.isnumeric(), txt.split(",")))
with open("workload2.txt", "w") as f:
    f.write(" ".join(workload))