from predictions import get_predictor

tp = get_predictor()
sites = tp.S["site"].index
print(len(sites), "known sites")
print(list(sites[:10]))                      # first 10 ids
print(tp.S["attrs"].head(10))                # their region, road type, location