import schrodinger as sc

domain = sc.Circle((0,0),1)

s = sc.Schrodinger2D(lambda x, y: 0, domain, gridSize=(30,30))

print(s.eigenvalues())
