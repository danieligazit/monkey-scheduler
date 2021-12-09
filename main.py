from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List

from google.protobuf.type_pb2 import Enum
from ortools.sat.python import cp_model

THURSDAY = 3
FRIDAY = 4
SATURDAY = 5
WEEKEND = [FRIDAY, SATURDAY]


def date_range(start_date: datetime, end_date: datetime):
    for n in range(int((end_date - start_date).days) + 1):
        yield start_date + timedelta(n)


class ShiftKind:
    WEEKDAY = 'Weekday'
    WEEKEND = 'Weekend'
    HOLIDAY = 'Holiday'


@dataclass(eq=True, unsafe_hash=True)
class Shift:
    start_date: datetime
    end: datetime
    kind: str


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
        for shift in all_shifts:
            num_shifts_worked += shifts[(monkey, shift)]

        model.Add(min_shifts_per_monkey <= num_shifts_worked)
        model.Add(num_shifts_worked <= max_shifts_per_monkey)

    # for monkey in all_monkeys:
    #     days_since_last_shift = []
    #
    #     for i, shift in enumerate(all_shifts):
    #         for previous_shift in all_shifts[:i]:
    #
    #             days_since_last_shift = model.NewIntVar(1, len(num_shifts))
    #             shifts[(monkey, shift)]
    #
    #             model.Maximize()
    #             days_since_last_shift += shifts[(monkey, shift)]

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
