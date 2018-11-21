from multiprocessing import Pool                                                
                                                                                
                                                                             
                                       
                                                                                
def run_process(cmd):                                                             
    os.system('python {}'.format(cmd))                                       
                                                                                
def f(x):
    print(x)
    for i in range(1000):
        x += i
    print(x, i)

    

if __name__ == '__main__':
    numGenerations = 100
    p = Pool(8)
    
    
    results = []
    results.append(p.apply_async(run_process, ('test4.py 1 2 3',)))
    results.append(p.apply_async(run_process, ('test4.py 4 5 6',)))
    results.append(p.apply_async(run_process, ('test4.py 7 8 9',)))
    for r in results:
        r.get()

    # results = p.apply_async(f, (1,))
    # results = p.apply_async(f, (5,))
    # results = p.apply_async(f, (2,))
    #.map(run_process, ['test4.py 1 2 3', 'test4.py a b c', 'test4.py what de fuck'])

                                                                    