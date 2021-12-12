from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import List

from ortools.sat.python import cp_model

THURSDAY = 3
FRIDAY = 4
SATURDAY = 5
WEEKEND = [FRIDAY, SATURDAY]


def date_range(start_date: datetime, end_date: datetime):
    for n in range(int((end_date - start_date).days) + 1):
        yield start_date + timedelta(n)


class ShiftKind(Enum):
    WEEKDAY = 'Weekday'
    WEEKEND = 'Weekend'
    HOLIDAY = 'Holiday'


@dataclass(eq=True, unsafe_hash=True)
class Shift:
    start_date: datetime
    end: datetime
    kind: ShiftKind


def create_shifts(start_date: datetime, end_date: datetime) -> List[Shift]:
    shifts = []

    for date in date_range(start_date, end_date):
        if date.weekday() == THURSDAY:
            shifts.append(Shift(
                start_date=date,
                end=date + timedelta(days=3),
                kind=ShiftKind.WEEKEND
            ))
            continue

        if date.weekday() in WEEKEND:
            continue

        shifts.append(Shift(
            start_date=date,
            end=date + timedelta(days=1),
            kind=ShiftKind.WEEKDAY
        ))

    return shifts


def negated_bounded_span(works, start, length):
    """Filters an isolated sub-sequence of variables assined to True.
  Extract the span of Boolean variables [start, start + length), negate them,
  and if there is variables to the left/right of this span, surround the span by
  them in non negated form.
  Args:
    works: a list of variables to extract the span from.
    start: the start to the span.
    length: the length of the span.
  Returns:
    a list of variables which conjunction will be false if the sub-list is
    assigned to True, and correctly bounded by variables assigned to False,
    or by the start or end of works.
  """
    sequence = []
    # Left border (start of works, or works[start - 1])
    if start > 0:
        sequence.append(works[start - 1])
    for i in range(length):
        sequence.append(works[start + i].Not())
    # Right border (end of works or works[start + length])
    if start + length < len(works):
        sequence.append(works[start + length])
    return sequence


def add_soft_sequence_constraint(model: cp_model.CpModel, works, hard_min, soft_min, min_cost,
                                 soft_max, hard_max, max_cost, prefix):
    """Sequence constraint on true variables with soft and hard bounds.
    This constraint look at every maximal contiguous sequence of variables
    assigned to true. If forbids sequence of length < hard_min or > hard_max.
    Then it creates penalty terms if the length is < soft_min or > soft_max.
    Args:
    model: the sequence constraint is built on this model.
    works: a list of Boolean variables.
    hard_min: any sequence of true variables must have a length of at least
      hard_min.
    soft_min: any sequence should have a length of at least soft_min, or a
      linear penalty on the delta will be added to the objective.
    min_cost: the coefficient of the linear penalty if the length is less than
      soft_min.
    soft_max: any sequence should have a length of at most soft_max, or a linear
      penalty on the delta will be added to the objective.
    hard_max: any sequence of true variables must have a length of at most
      hard_max.
    max_cost: the coefficient of the linear penalty if the length is more than
      soft_max.
    prefix: a base name for penalty literals.
    Returns:
    a tuple (variables_list, coefficient_list) containing the different
    penalties created by the sequence constraint.
    """
    cost_literals = []
    cost_coefficients = []

    spans = []
    for length in range(len(works))[4:5]:
        for start in range(len(works) - length + 1):
            span = negated_bounded_span(works, start, length)
            name = f': over_span(start={start}, length={length})'
            span = model.New(span)

            spans.append(span)
    print(spans)
    model.Maximize(sum(spans))


def add_sequence_constraint(model, works, hard_min):
    # Forbid sequences that are too short.
    for length in range(1, hard_min):
        for start in range(len(works) - length - 1):
            print(span)
            model.AddBoolOr(negated_bounded_span(works, start, length))


def main():
    all_shifts = create_shifts(start_date=datetime(2022, 1, 2), end_date=(datetime(2022, 2, 28)))
    num_shifts = len(all_shifts)
    all_monkeys = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    num_monkeys = len(all_monkeys)

    model = cp_model.CpModel()

    shifts = {}

    for monkey in all_monkeys:
        for shift in all_shifts:
            shifts[(monkey, shift)] = model.NewBoolVar(f'shift_{monkey}_{shift}')

    # Each shift is assigned to exactly one monkey.
    for shift in all_shifts:
        model.Add(sum(shifts[(monkey, shift)] for monkey in all_monkeys) == 1)

    # A monkey should not be assigned to two shifts in a row
    for previous_shift, shift in zip(all_shifts, all_shifts[1:]):
        for monkey in all_monkeys:
            model.Add(shifts[(monkey, shift)] + shifts[(monkey, previous_shift)] < 2)

    # Try to distribute the shifts evenly, so that each nurse works
    # min_shifts_per_nurse shifts. If this is not possible, because the total
    # number of shifts is not divisible by the number of nurses, some nurses will
    # be assigned one more shift.
    min_shifts_per_monkey = num_shifts // num_monkeys
    if num_shifts % num_monkeys == 0:
        max_shifts_per_monkey = min_shifts_per_monkey
    else:
        max_shifts_per_monkey = min_shifts_per_monkey + 1

    for monkey in all_monkeys:
        num_shifts_worked = 0
        for index, shift in enumerate(all_shifts):
            num_shifts_worked += shifts[(monkey, shift)]

        model.Add(min_shifts_per_monkey <= num_shifts_worked)
        model.Add(num_shifts_worked <= max_shifts_per_monkey)

    for monkey in all_monkeys:
        works = [shifts[monkey, shift] for shift in all_shifts]

        add_sequence_constraint(model, works, 1)
        hard_min, soft_min, min_cost, soft_max, hard_max, max_cost = (1, 2, 20, 3, 4, 5)

        add_soft_sequence_constraint(
            model, works, hard_min, soft_min, min_cost, soft_max, hard_max,
            max_cost, f'shift_constraint(employee {monkey}, shift {shift})')

        # days_since_last_shift = []
        #
        # for i, shift in enumerate(all_shifts):
        #     for previous_shift in all_shifts[:i]:
        #         days_since_last_shift = model.NewIntVar(1, len(num_shifts))
        #         shifts[(monkey, shift)]
        #
        #         model.Maximize()
        #         days_since_last_shift += shifts[(monkey, shift)]

    # model.Maximize(
    #     sum(shift_requests[n][d][s] * shifts[(n, d, s)] for n in all_nurses
    #         for d in all_days for s in all_shifts))

    # Creates the solver and solve.
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL:
        print('Solution:')
        for shift in all_shifts:
            for monkey in all_monkeys:
                if not solver.Value(shifts[(monkey, shift)]) == 1:
                    continue

                print(shift, monkey)

                # if shift_requests[n][d][s] == 1:
                #     print('Nurse', n, 'works shift', s, '(requested).')
                # else:
                #     print('Nurse', n, 'works shift', s,
                #           '(not requested).')
        # print(f'Number of shift requests met = {solver.ObjectiveValue()}',
        #       f'(out of {num_nurses * min_shifts_per_monkey})')
    else:
        print('No optimal solution found !')

    # Statistics.
    print('\nStatistics')
    print('  - conflicts: %i' % solver.NumConflicts())
    print('  - branches : %i' % solver.NumBranches())
    print('  - wall time: %f s' % solver.WallTime())


if __name__ == '__main__':
    main()
