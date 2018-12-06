from ElaphurusPGM.Model import BayesianModel
from ElaphurusPGM.VE import VariableElimination
from ElaphurusPGM.CPD import CPD
from ElaphurusPGM.RejectSampling import BayesianModelSampling

model = BayesianModel([('B','A'),('E','A'),('A','J'),('A','M')])

# Defining the CPDs:
cpd_b = CPD('B',2,[[0.001,0.999]])
cpd_e = CPD('E',2,[[0.002,0.998]])
cpd_a = CPD('A',2,[[0.95,0.94,0.29,0.001],[0.05,0.06,0.71,0.999]],
                   parent=['B','E'],parent_card=[2,2])
cpd_j = CPD('J',2,[[0.90,0.05],[0.10,0.95]],
                   parent=['A'],parent_card=[2])
cpd_m = CPD('M',2,[[0.70,0.01],[0.30,0.99]],
                   parent=['A'],parent_card=[2])

# # Associating the CPDs with the network structure.
model.add_cpds(cpd_b, cpd_e, cpd_a,cpd_j,cpd_m)

# # Some other methods
# print(model.get_cpds())
print("exact inference")
infer = VariableElimination(model)

res, order = infer.query(['A'])
print(res['A'].values[0])
print("elimination order", order)

res, order = infer.query(['J','M'])
print(res['JDP'].values[0][1])
print("elimination order", order)

res, order = infer.query(['A'],evidence={'J':0,'M':1})
print(res['A'].values[0])
print("elimination order", order)

res, order = infer.query(['B'],evidence={'A':0})
print(res['B'].values[0])
print("elimination order", order)

res, order = infer.query(['B'],evidence={'J':0,'M':1})
print(res['B'].values[0])
print("elimination order", order)

res, order = infer.query(['J','M'],evidence={'B':1})
print(res['JDP'].values[0][1])
print("elimination order", order)

print("\nreject sample")
BMSampling = BayesianModelSampling(model)

sampled = BMSampling.reject_sample(10000)
print(len(sampled[sampled["A"] == 0])*1.0/len(sampled))

sampled = BMSampling.reject_sample(20000)
print(len(sampled[(sampled["J"] == 0) & (sampled["M"] == 1)])*1.0/len(sampled))

sampled = BMSampling.reject_sample(2000,evidence={'J':0,'M':1})
print(len(sampled[sampled["A"] == 0])*1.0/len(sampled))

sampled = BMSampling.reject_sample(1000,evidence={'A':0})
print(len(sampled[sampled["B"] == 0])*1.0/len(sampled))

sampled = BMSampling.reject_sample(10000,evidence={'J':0,'M':1})
print(len(sampled[sampled["B"] == 0])*1.0/len(sampled))

sampled = BMSampling.reject_sample(2000,evidence={'B':1})
print(len(sampled[(sampled["J"] == 0) & (sampled["M"] == 1)])*1.0/len(sampled))
