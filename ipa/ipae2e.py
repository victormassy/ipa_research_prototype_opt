"""
IPA End-to-end MPC in MP-SPDZ
"""
import operator
from enum import IntEnum

from Compiler.library import print_ln, tree_reduce, for_range_opt, for_range
from Compiler.types import Array, Matrix, sint, sintbit

from sort import radix_sort

def print_reports_row(data, i):
    print_ln('%s %s %s %s %s %s', data[i][0].reveal(), data[i][1].reveal(), data[i][2].reveal(), data[i][3].reveal())

def print_reports(reports, nb_rows):
    @for_range_opt(nb_rows)
    def _(i):
        print_reports_row(reports,i)

class Columns(IntEnum):
    """
    These are the Columns in the reports Matrix
    """

    MATCHKEY = 0
    IS_TRIGGER = 1
    VALUE = 2
    BREAKDOWN_KEY = 3


def load_data(nb_rows):   
    reports = Matrix(nb_rows, 4, sint)
    match_keys = Array(nb_rows, sint)
    print_ln("I'm loading data")
    print_ln("nb rows %s", nb_rows)
    @for_range_opt(nb_rows)
    def _(i): 
            #For each field in the report 
            for j in range(4): 
                #Read input from different parties 
                for k in range(3):
                    #Sum shares    
                    reports[i][j] += sint.get_input_from(k) 
    match_keys = reports.get_column(Columns.MATCHKEY)
    #reports.print_reveal_nested()
    return reports, match_keys

"""
def load_data(numrows):
    

    reports = Matrix(numrows, 4, sint)
    reports.assign_vector(sint.get_input_from(0, size=numrows * 4))
    match_keys = reports.get_column(Columns.MATCHKEY)
    return reports, match_keys
"""




def oblivious_attribution(
    reports: Matrix,
    breakdown_values: int,
) -> Array:
    """
    Perform the oblivious attribution, capping and aggregation
    """

    numrows, _ = reports.sizes

    # Edge cases: imagine that the match-keys are extended at either
    # end with keys that are different than any of the real keys
    # That means that helperbits[0] = 0
    # and that stopbits at the end are zeroed out
    helperbits = Array(numrows, sintbit)
    match_keys = reports.get_column(Columns.MATCHKEY)

    helperbits.assign_vector(
        match_keys.get_vector(size=numrows - 1)
        == match_keys.get_vector(base=1, size=numrows - 1),
        base=1,
    )
    helperbits[0] = 0
    # helperbits[idx] = 1 if there's a transition to a new match key

    # is_trigger[idx] = 1  means this is a trigger event
    # we want to match all non-trigger events (i.e. source events)
    # with trigger events with the same matchkey

    is_trigger = Array(numrows, sintbit)
    is_trigger.assign_vector(reports.get_column(Columns.IS_TRIGGER))
    helperbits_and_istrigger = helperbits.get_vector() & is_trigger.get_vector()

    # Initialize for results after the first pass
    stopbit = Array(numrows, sintbit)
    credit = Array(numrows, sint)
    repval = reports.get_column(Columns.VALUE)

    stopbit.assign_vector(helperbits_and_istrigger.get_vector(base=1, size=numrows - 1))
    stopbit[numrows - 1] = 0
    credit[numrows - 1] = repval[numrows - 1]

    credit.assign_vector(
        repval.get_vector(size=numrows - 1)
        + stopbit.get_vector(size=numrows - 1)
        * repval.get_vector(base=1, size=numrows - 1)
    )

    zeros = Array(numrows // 2, sintbit)
    zeros.assign_all(0)

    stepsize = 1

    # compute the oblivious "tree" attribution algorithm
    while stepsize < numrows // 2:
        stepsize *= 2

        new_size = numrows - stepsize

        flag = stopbit.get_vector(size=new_size) & helperbits_and_istrigger.get_vector(
            base=stepsize, size=new_size
        )
        new_credit = credit.get_vector(size=new_size) + flag * credit.get_vector(
            base=stepsize, size=new_size
        )
        stopbit.assign_vector(flag & stopbit.get_vector(base=stepsize, size=new_size))

        stopbit.assign_vector(zeros.get_vector(size=stepsize), base=new_size)

        # Replace the first new_size elements, leaving the others alone
        credit.assign_vector(new_credit)

    # Calculate final_credits of source events by zering out the values of trigger rows.
    final_credits = Array(numrows, sint)
    final_credits.assign_vector((1 - is_trigger.get_vector()) * credit.get_vector())
    return helperbits, final_credits


def sequential_capping(numrows, final_credits, helperbits):
    # CAPPING
    # (SEQUENTIAL CAPPING ALGORITHM)
    # there is a known bug in this version of capping;
    # we used the parallel version for PATCG benchmarks
    print_ln("sequential capping")
    current_contribution = Array(numrows, sint)
    current_contribution.assign_vector(0)
    cap = 10
    rows = range(numrows)
    @for_range_opt(numrows)
    def _(row):
        current_contribution[row] = current_contribution[row] * helperbits[row]

        _min = (final_credits[row] < cap - current_contribution[row]).if_else(
            final_credits[row], cap - current_contribution[row]
        )

        final_credits[row] = (current_contribution[row] <= cap) * _min
        current_contribution[row] = current_contribution[row] + final_credits[row]

    return final_credits


def parallel_capping(numrows, final_credits, helperbits):
    # PARALLEL CAPPING ALGORITHM
    
    stopbit = Array(numrows, sint)

    zeros = Array(numrows // 2, sintbit)
    zeros.assign_all(0)

    stopbit.assign_vector(zeros.get_vector() + 1)
    current_contribution = Array(numrows, sint)
    current_contribution.assign_vector(final_credits.get_vector())
    cap = 10

    stepsize = 1
    
    while stepsize < numrows // 2:
        stepsize *= 2

        new_size = numrows - stepsize

        flag = stopbit.get_vector(size=new_size) * helperbits.get_vector(
            base=stepsize, size=new_size
        )
        new_current_contribution = current_contribution.get_vector(
            size=new_size
        ) + flag * current_contribution.get_vector(base=stepsize, size=new_size)
        stopbit.assign_vector(flag * stopbit.get_vector(base=stepsize, size=new_size))

        # Replace the first new_size elements, leaving the others alone
        current_contribution.assign_vector(new_current_contribution)
    
    compare_bit = current_contribution.get_vector() <= cap
    intermediary = cap - helperbits.get_vector(base=1, size=numrows - 1) * (
        cap
        + compare_bit.get_vector(base=1, size=numrows - 1)
        * (
            (
                cap - current_contribution.get_vector(base=1, size=numrows - 1)
            ).get_vector()
        )
    )

    final_credits.assign_vector(
        intermediary.get_vector()
        + compare_bit.get_vector(base=1, size=numrows - 1)
        * (
            final_credits.get_vector(base=1, size=numrows - 1)
            - intermediary.get_vector()
        ),
        base=1,
    )

    return final_credits


def aggregate(reports, breakdown_values, final_credits):
    # AGGREGATION
    breakdown = reports.get_column(Columns.BREAKDOWN_KEY)

    breakdown_keys = range(breakdown_values)
    
    
    # One can use sum, but tree_reduce appears to be more efficient
    return Array(breakdown_values, sint).create_from(
        [
            tree_reduce(operator.add, (breakdown == breakdown_key) * final_credits)
            for breakdown_key in breakdown_keys
        ]
    )
    
    
def aggregate_opt(reports, breakdown_values, final_credits, sort_function):
    # AGGREGATION
    breakdown = reports.get_column(Columns.BREAKDOWN_KEY)
    numrows, _ = reports.sizes
    
    from sort import sort_functions
    from Compiler.library import print_ln_if, do_while

    numrows, _ = reports.sizes
    matrix_value = Matrix(numrows, 2, sint)
    
    matrix_value.set_column(0, breakdown)
    matrix_value.set_column(1, final_credits.get_vector())
    sort_function(breakdown, matrix_value)

    """
    prev_key = matrix_value[0][0]
    sum = 0
    for i in range(numrows-1):
        curr_key = matrix_value[i][0]
        next_key = matrix_value[i+1][0]
        
        same_key = (prev_key == curr_key)
        
        sum = sum*same_key + matrix_value[i][1]
        matrix_value[i][1]  = sum  
        same_key = (curr_key != next_key)
        print_ln_if(same_key.reveal(),"%s %s", curr_key.reveal(), sum.reveal())
        prev_key = curr_key  

    print_ln("%s %s", matrix_value[numrows-1][0].reveal(), sum.reveal())
    """ 
    
    @for_range_opt(numrows-1)
    def _(i):
        @do_while
        def _():
            curr_key = matrix_value[i][0]
            next_key = matrix_value[i+1][0]
            same_key = (next_key == curr_key)*(i<numrows-2)
            matrix_value[i+1][1]  = same_key*matrix_value[i][1] + matrix_value[i+1][1] 
            i.update(i+1)
            return (same_key.reveal())
        print_ln("%s %s", matrix_value[i-1][0].reveal(), matrix_value[i-1][1].reveal())
   
