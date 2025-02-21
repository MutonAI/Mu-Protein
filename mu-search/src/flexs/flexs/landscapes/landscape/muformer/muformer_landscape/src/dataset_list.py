
class MutationDatasets():
    def __init__(self):
        self.deep_sequence_collection = [
            'AMIE_PSEAE', 'B3VI55_LIPST', 'B3VI55_LIPSTSTABLE', 'BF520_ENV', 'BG505_ENV',
            'BG_STRSQ', 'BLAT_ECOLX_Ostermeier2014', 'BLAT_ECOLX_Palzkill2012', 
            'BLAT_ECOLX_Ranganathan2015', 'BLAT_ECOLX_Tenaillon2013', 'BRCA1_HUMAN_BRCT', 
            'BRCA1_HUMAN_RING', 'CALM1_HUMAN', 'DLG4_RAT_Ranganathan2012', 'GAL4', 
            'HG_FLU', 'HIS7_YEAST','HSP82', 'IF1_ECOLI', 
            'KKA2', 'MK01_HUMAN', 'MTH3_HAEAESTABILIZED', 'P84126_THETH', 
            'PABP_doubles', 'PABP_singles', 'PA_FLU', 'POLG_HCVJF', 
            'PTEN_HUMAN', 'RASH_HUMAN', 'RL401_YEAST_Bolon2013', 'RL401_YEAST_Bolon2014', 
            'RL401_YEAST_Fraser2016', 'SUMO1_HUMAN', 'TIM_SULSO', 'TIM_THEMA', 
            'TPK1_HUMAN', 'TPMT_HUMAN', 'UBC9_HUMAN', 
            'UBE4B', 'YAP1_HUMAN_Fields2012-singles', 'parEparD_Laub2015_all'
        ]
        
        self.envision_collection = [
        'Aminoglycoside', 
        'BRCA1_BARD1',
        'BRCA1_E3',
        'HSP90',
        'PSD95',
        'Pab1',
        'Protein_G',
        'TEM-1',
        'UBE4B',
        'Ubiquitin',
        'Ubiquitin_E1',
        'Yap65',
        ]
        
        self.flip_collection = [
            'AAV','GB1'
        ]
        
        self.tranception_substitution_collection = [
            'A0A140D2T1_ZIKV_Sourisseau_growth_2019','A0A192B1T2_9HIV1_Haddox_2018','A0A1I9GEU1_NEIME_Kennouche_2019',
            'A0A2Z5U3Z0_9INFA_Doud_2016','A0A2Z5U3Z0_9INFA_Wu_2014','A4D664_9INFA_Soh_CCL141_2019','A4GRB6_PSEAI_Chen_2020',
            'A4_HUMAN_Seuma_2021','AACC1_PSEAI_Dandage_2018','ADRB2_HUMAN_Jones_2020','AMIE_PSEAE_Wrenbeck_2017',
            'B3VI55_LIPST_Klesmith_2015','BLAT_ECOLX_Deng_2012','BLAT_ECOLX_Firnberg_2014','BLAT_ECOLX_Jacquier_2013',
            'BLAT_ECOLX_Stiffler_2015','BRCA1_HUMAN_Findlay_2018','C6KNH7_9INFA_Lee_2018','CALM1_HUMAN_Weile_2017',
            'CAPSD_AAV2S_Sinai_substitutions_2021','CCDB_ECOLI_Adkar_2012','CCDB_ECOLI_Tripathi_2016',
            'CP2C9_HUMAN_Amorosi_abundance_2021','CP2C9_HUMAN_Amorosi_activity_2021','DLG4_HUMAN_Faure_2021',
            'DLG4_RAT_McLaughlin_2012','DYR_ECOLI_Thompson_plusLon_2019','ENV_HV1B9_DuenasDecamp_2016',
            'ENV_HV1BR_Haddox_2016','ESTA_BACSU_Nutschel_2020','F7YBW8_MESOW_Aakre_2015','GAL4_YEAST_Kitzman_2015',
            'GCN4_YEAST_Staller_induction_2018','GFP_AEQVI_Sarkisyan_2016','GRB2_HUMAN_Faure_2021','HIS7_YEAST_Pokusaeva_2019',
            'HSP82_YEAST_Flynn_2019','HSP82_YEAST_Mishra_2016','I6TAH8_I68A0_Doud_2015','IF1_ECOLI_Kelsic_2016',
            'KCNH2_HUMAN_Kozek_2020','KKA2_KLEPN_Melnikov_2014','MK01_HUMAN_Brenan_2016','MSH2_HUMAN_Jia_2020','MTH3_HAEAE_Rockah-Shmuel_2015',
            'NCAP_I34A1_Doud_2015','NRAM_I33A0_Jiang_standard_2016','NUD15_HUMAN_Suiter_2020','P53_HUMAN_Giacomelli_NULL_Etoposide_2018',
            'P53_HUMAN_Giacomelli_NULL_Nutlin_2018','P53_HUMAN_Giacomelli_WT_Nutlin_2018','P53_HUMAN_Kotler_2018','P84126_THETH_Chan_2017',
            'PABP_YEAST_Melamed_2013','PA_I34A1_Wu_2015','POLG_CXB3N_Mattenberger_2021','POLG_HCVJF_Qi_2014','PTEN_HUMAN_Matreyek_2021',
            'PTEN_HUMAN_Mighell_2018','Q2N0S5_9HIV1_Haddox_2018','Q59976_STRSQ_Romero_2015','R1AB_SARS2_Flynn_growth_2022',
            'RASH_HUMAN_Bandaru_2017','REV_HV1H2_Fernandes_2016','RL401_YEAST_Mavor_2016','RL401_YEAST_Roscoe_2013','RL401_YEAST_Roscoe_2014',
            'SC6A4_HUMAN_Young_2021','SCN5A_HUMAN_Glazer_2019','SPG1_STRSG_Olson_2014','SPIKE_SARS2_Starr_bind_2020','SPIKE_SARS2_Starr_expr_2020',
            'SRC_HUMAN_Ahler_CD_2019','SUMO1_HUMAN_Weile_2017','SYUA_HUMAN_Newberry_2020','TADBP_HUMAN_Bolognesi_2019','TAT_HV1BR_Fernandes_2016',
            'TPK1_HUMAN_Weile_2017','TPMT_HUMAN_Matreyek_2018','TPOR_HUMAN_Bridgford_S505N_2020','TRPC_SACS2_Chan_2017','TRPC_THEMA_Chan_2017',
            'UBC9_HUMAN_Weile_2017','UBE4B_MOUSE_Starita_2013','VKOR1_HUMAN_Chiasson_abundance_2020','VKOR1_HUMAN_Chiasson_activity_2020','YAP1_HUMAN_Araya_2012'
            ]
        
        self.tranception_indel_collection = [
            'A0A1J4YT16_9PROT_Davidi_2020', 'B1LPA6_ECOSM_Russ_2020',
            'BLAT_ECOLX_Gonzalez_indels_2019', 'CAPSD_AAV2S_Sinai_indels_2021',
            'HIS7_YEAST_Pokusaeva_indels_2019','PTEN_HUMAN_Mighell_deletions_2018',
            'P53_HUMAN_Kotler_deletions_2018'
        ]

        self.data_collection_dict = {
            'DeepSequence': self.deep_sequence_collection,
            'Envision': self.envision_collection,
            'Flip': self.flip_collection,
            'TranceptionIndels':self.tranception_indel_collection,
            'TranceptionSubs':self.tranception_substitution_collection,
        }
        
        self.data_foldpath_map = {
            'DeepSequence':'DeepSequenceDataSet',
            'Envision':'EnvisionDataSet',
            'Flip':'FlipDataSet',
            'TranceptionIndels':'TranceptionIndelsDataSet',
            'TranceptionSubs':'TranceptionSubsDataSet',
        }

        self.prefered_batch_size = {
            'BF520_ENV': 4, 
            'BG505_ENV': 4,
            'PABP_doubles': 6,
            'BRCA1_HUMAN_BRCT': 4, 
            'BRCA1_HUMAN_RING': 4, 
            'HIS7_YEAST': 32, 
            'POLG_HCVJF': 6, 
            'UBE4B': 2,
            'AAV':{
                'one_vs_many':4,
            },
            'GB1':{
                'three_vs_rest':8,
                'low_vs_high':16,
            },
            'CALM1_HUMAN':32,
            'MK01_HUMAN':12,
            'A0A1J4YT16_9PROT_Davidi_2020':4,
        }
    
    def get_data_list(self, data_source):
        return self.data_collection_dict[data_source]
    
    def get_data_foldpath(self, data_source):
        return self.data_foldpath_map[data_source]