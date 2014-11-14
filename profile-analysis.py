import pstats
p = pstats.Stats('profile.stats')
p.strip_dirs().sort_stats('time').print_stats()
