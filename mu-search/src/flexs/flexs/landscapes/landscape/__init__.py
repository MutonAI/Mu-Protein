# import os
# import importlib
# from muformer_landscape.landscape import MuformerLandscape, MuformerLandscapeList

# protein_alphabet = 'ACDEFGHIKLMNPQRSTVWY'

# task_alphabet_dict = {
#     'TEM': protein_alphabet,
# }

# task_wild_type_dict = {
#     'TEM': 'MSIQHFRVALIPFFAAFCLPVFAHPETLVKVKDAEDQLGARVGYIELDLNSGKILESFRPEERFPMMSTFKVLLCGAVLSRVDAGQEQLGRRIHYSQNDLVEYSPVTEKHLTDGMTVRELCSAAITMSDNTAANLLLTTIGGPKELTAFLHNMGDHVTRLDRWEPELNEAIPNDERDTTMPAAMATTLRKLLTGELLTLASRQQLIDWMEADKVAGPLLRSALPAGWFIADKSGAGERGSRGIIAALGPDGKPSRIVVIYTTGSQATMDERNRQIAEIGASLIKHW',
# }

# landscape_collection = {'muformer': MuformerLandscapeList}

# def get_landscape(args):
#     if args.landscape == 'muformer':
#         landscape = landscape_collection[args.landscape](args, task_wild_type_dict[args.task])
#     else:
#         raise NotImplementedError
#     return landscape, task_alphabet_dict[args.task], task_wild_type_dict[args.task]

