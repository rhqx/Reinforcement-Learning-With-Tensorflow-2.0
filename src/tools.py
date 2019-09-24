import itertools

class HyperParam:
    """
    input:
        param_list: [[name1, parameter_list_1], ...}
    """
    def __init__(self, *argv):
        combs = []
        for param_list in argv:
            self.names = [i[0] for i in param_list]
            params = [i[1] for i in param_list]
            combs.append(itertools.product(*params)) 
        self.comb = itertools.chain(*combs)

    def next(self):
        try:
            ins = next(self.comb)
        except:
            return False 
        ins_str = '_'.join([str(i) for i in ins])
        return (ins, ins_str)

    def names(self):
        return self.names


if __name__ == "__main__":
    p1 = [
        ['a',[1,2]],
        ['b',[3,4]]
    ]

    p2 = [
        ['a',[5,6]],
        ['b',[7]]
    ]

    hp = HyperParam(p1)
    while True:
        a = next(hp)
        if not a:
            break
        print(a)


