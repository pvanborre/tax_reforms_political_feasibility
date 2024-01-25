import numpy
import pandas

import click

from openfisca_core.simulation_builder import SimulationBuilder
from openfisca_france import FranceTaxBenefitSystem

pandas.options.display.max_columns = None



def initialize_simulation(tax_benefit_system, data_persons):
    """
    Declares all 4 types of OpenFisca : individuals, households, families, foyer_fiscaux
    see https://openfisca.org/doc/simulate/run-simulation.html (part run simulation on data)
    """

    sb = SimulationBuilder()
    sb.create_entities(tax_benefit_system)

    # id are variables idmen (household), idfoy (foyer fiscal), idfam (family)
    # roles within these entities : quimen, quifoy et quifam 

    # individuals
    sb.declare_person_entity('individu', data_persons.noindiv)

    # households 
    build_entity(data_persons, sb, nom_entite = "menage", nom_entite_pluriel = "menages", id_entite = "idmen", id_entite_join = "idmen_original",
                   role_entite = "quimen", nom_role_0 = "personne_de_reference", nom_role_1 = "conjoint", nom_role_2 = "enfant")
    
    # foyers fiscaux
    build_entity(data_persons, sb, nom_entite = "foyer_fiscal", nom_entite_pluriel = "foyers fiscaux", id_entite = "idfoy", id_entite_join = "idfoy",
                   role_entite = "quifoy", nom_role_0 = "declarant_principal", nom_role_1 = "conjoint", nom_role_2 = "personne_a_charge")
    
    # families
    build_entity(data_persons, sb, nom_entite = "famille", nom_entite_pluriel = "familles", id_entite = "idfam", id_entite_join = "idfam",
                   role_entite = "quifam", nom_role_0 = "demandeur", nom_role_1 = "conjoint", nom_role_2 = "enfant")

    simulation = sb.build(tax_benefit_system)
    return simulation


def build_entity(data_persons, sb, nom_entite, nom_entite_pluriel, id_entite, id_entite_join, role_entite, nom_role_0, nom_role_1, nom_role_2):
    """
    Recodes roles within an entity and associates in the simulation everyone to its role
    """

    # requires the good number of entities : .unique()
    # otherwise OpenFisca believes that same number of entities (households, foyer fiscaux, families) than individuals 
    instance = sb.declare_entity(nom_entite, data_persons[id_entite].unique())

    print("nombre de " + nom_entite_pluriel, instance.count)
    print("rôles acceptés par OpenFisca pour les " + nom_entite_pluriel, instance.entity.flattened_roles)


    # join_with_persons function need as input a string array so we recode it 
    data_persons[role_entite] = numpy.select(
        [data_persons[role_entite] == 0, data_persons[role_entite] == 1, data_persons[role_entite] == 2],
        [nom_role_0, nom_role_1, nom_role_2],
        default="anomalie"  
    )

    # We associate every individual to its role in the entity 
    sb.join_with_persons(instance, data_persons[id_entite_join], data_persons[role_entite])

    # Check that roles are in the scope of the roles we expected  
    print("rôles de chacun dans son " + nom_entite, instance.members_role)
    assert("anomalie" not in instance.members_role)

    print("\n\n\n\n\n")





def deal_with_married_couples(data_people):
    """
    Here are the statut_marital values: (see https://github.com/openfisca/openfisca-france-data/blob/6f7af1a194baf07ab4cfacb6ce1d4e817d9913b8/openfisca_france_data/erfs_fpr/input_data_builder/step_03_variables_individuelles.py#L929 )
      1 - "Marié",
      2 - "Célibataire",
      3 - "Divorcé",
      4 - "Veuf"

    The goal of this function is to perform equal split of earnings between couples (equal split couples)
    """
    earnings_variables = ["chomage_brut", "pensions_alimentaires_percues", "pensions_invalidite",
                      "rag", "retraite_brute", "ric", "rnc", "rpns_imposables", "salaire_de_base",
                      "primes_fonction_publique", "traitement_indiciaire_brut"]

    married_df = data_people[data_people['statut_marital'] == 1]
    married_mean_df = married_df.groupby('idfoy')[earnings_variables].mean().reset_index()
    married_augmented_df = pandas.merge(married_mean_df, married_df, on='idfoy', suffixes=('', '_oldvalues'))

    columns_to_drop = married_augmented_df.filter(like='_oldvalues').columns
    married_augmented_df.drop(columns=columns_to_drop, inplace=True)

    single_df = data_people[data_people['statut_marital'] != 1]
    final_df = pandas.concat([single_df, married_augmented_df]).sort_values(by='idfoy')

    print("final_df", final_df)
    return final_df




@click.command()
@click.option('-y', '--beginning_year', default = None, type = int, required = True)
@click.option('-e', '--end_year', default = -1, type = int, required = True)
def simulate_without_reform(beginning_year = None, end_year = None):
    if end_year == -1:
        end_year = beginning_year + 1 #reform phased in over 2 years only 

    # load individuals and households data and combine the two datasets  
    filename = "../data/{}/openfisca_erfs_fpr_{}.h5".format(beginning_year, beginning_year)
    data_people_brut = pandas.read_hdf(filename, key = "individu_{}".format(beginning_year))
    data_households_brut =  pandas.read_hdf(filename, key = "menage_{}".format(beginning_year))
    data_people = data_people_brut.merge(data_households_brut, right_index = True, left_on = "idmen", suffixes = ("", "_x"))

    # check whether really needed here i don't think so really at the individual level not the foyer fiscal level
    # Weight adjustment : wprm weights are for households, whereas we work on foyers fiscaux
    # the idea in OpenFisca France-data is to say that individuals have the weight of their households (what is done in the left join above)
    # and then summing over individuals of the foyer fiscal gives the weight of the foyer fiscal
    sum_wprm_by_idfoy = data_people.groupby('idfoy')['wprm'].sum().reset_index()
    sum_wprm_by_idfoy = sum_wprm_by_idfoy.rename(columns={'wprm': 'weight_foyerfiscal'})
    data_people = pandas.merge(data_people, sum_wprm_by_idfoy, on='idfoy')



    # perform equal split of earnings within couples 
    data_people = deal_with_married_couples(data_people)

    print("People data")
    print(data_people, "\n\n\n\n\n")

    
    #####################################################
    ########### Simulation ##############################
    #####################################################

    tax_benefit_system = FranceTaxBenefitSystem()
    simulation = initialize_simulation(tax_benefit_system, data_people)
    beginning_reform = str(beginning_year)
    end_reform = str(end_year) 
    print("Years under consideration", beginning_reform, end_reform)

    data_households = data_people.drop_duplicates(subset='idmen', keep='first')

    """
    Here we use the same simulation, that is same people with same variables
    And we compute its tax for another year 
    so that we can deduce T_after_reform(y) - T_before_reform(y)

    The only thing we need to do is to change y to get y_hat (account for inflation) : T_after_reform(y_hat) - T_before_reform(y)
    """

    # data we can get from INSEE website https://www.insee.fr/fr/statistiques/serie/001763852
    # check these values again : I took only January values 
    CPI = {'2019': 102.67, '2018': 101.67, '2017': 100.41, '2016': 99.07, '2015': 98.85, '2014': 99.26, '2013': 98.71, '2012': 97.68, '2011': 95.51, '2010': 93.92, '2009': 92.98, '2008': 92.33, '2007': 89.85, '2006': 88.73, '2005': 86.91, '2004': 85.61, '2003': 84.42, '2002': 82.89}
    inflation_coeff = (CPI[end_reform]-CPI[beginning_reform])/CPI[beginning_reform]
    print("Inflation coefficient", inflation_coeff)

    earnings_columns = ["chomage_brut", "pensions_alimentaires_percues", "pensions_invalidite", "rag", "retraite_brute",
                "ric", "rnc", "rpns_imposables", "salaire_de_base", "primes_fonction_publique", "traitement_indiciaire_brut"]


    for ma_variable in data_people.columns.tolist():
        if ma_variable not in ["idfam", "idfoy", "idmen", "noindiv", "quifam", "quifoy", "quimen", "wprm", "prest_precarite_hand",
                            "taux_csg_remplacement", "idmen_original", "idfoy_original", "idfam_original",
                            "idmen_original_x", "idfoy_original_x", "idfam_original_x", "weight_foyerfiscal", "prest_precarite_hand",
                            "idmen_x", "idfoy_x", "idfam_x"]: #variables that cannot enter in a simulation (id, roles, weights...)
            
            if ma_variable not in ["loyer", "zone_apl", "statut_occupation_logement", "taxe_habitation", "logement_conventionne"]: # all variables at the individual level
                simulation.set_input(ma_variable, beginning_reform, numpy.array(data_people[ma_variable]))

                if ma_variable not in earnings_columns: #variables where no need to account for inflation
                    simulation.set_input(ma_variable, end_reform, numpy.array(data_people[ma_variable])) 
                else: #here we account for inflation
                    simulation.set_input(ma_variable, end_reform, numpy.array(data_people[ma_variable])*(1+inflation_coeff))

            else: # all variables at the household level
                simulation.set_input(ma_variable, beginning_reform, numpy.array(data_households[ma_variable]))
                simulation.set_input(ma_variable, end_reform, numpy.array(data_households[ma_variable]))
    


    total_taxes_before_reform = simulation.calculate('impot_revenu_restant_a_payer', beginning_reform)
    print(total_taxes_before_reform)
    total_taxes_after_reform = simulation.calculate('impot_revenu_restant_a_payer', end_reform)
    print(total_taxes_after_reform)

    tax_difference = total_taxes_after_reform - total_taxes_before_reform
    data_people['tax_difference'] = tax_difference

    # in our data we do not have capital revenue (rvcm) so we rank people according to their normal income
    data_people['total_earning'] = data_people[earnings_columns].sum(axis=1)
    data_people['earnings_rank'] = data_people['total_earning'].rank().astype(int)
    print("data_people", data_people)

    data_people.to_csv(f'excel/{beginning_reform}-{end_reform}/people_{beginning_reform}-{end_reform}.csv', index=False)







simulate_without_reform()