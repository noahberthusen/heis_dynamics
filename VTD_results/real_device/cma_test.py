import cma
import pickle

es = pickle.load(open('optimizer_dump', 'rb'))
print(es.result_pretty())


"""
t = 6
final/bestever f-value = 2.105091e-02 5.589778e-03
incumbent solution: [1.390959928225458, 5.161853458469686, 4.843828380424557, -2.176156970797431]
std deviation: [0.08567719880667704, 0.06081922342565176, 0.07917843394588359, 0.06477152638029067]

final/bestever f-value = 4.890205e-03 4.890205e-03
incumbent solution: [1.683814354167617, 5.222804166636025, 5.230601344024533, -2.3336124354750734]
std deviation: [0.1129771824905318, 0.06750349208133614, 0.10121326355155201, 0.07759368456095002]

"""