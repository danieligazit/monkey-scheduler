from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Dict, Tuple, Iterator

from ortools.sat.python import cp_model
from ortools.sat.python.cp_model import IntVar, CpModel, CpSolver

THURSDAY = 3
FRIDAY = 4
SATURDAY = 5
WEEKEND = [FRIDAY, SATURDAY]
Monkey = str


class ShiftKind(Enum):
    WEEKDAY = 'Weekday'
    WEEKEND = 'Weekend'
    HOLIDAY = 'Holiday'


@dataclass(eq=True, unsafe_hash=True)
class Shift:
    start_date: datetime
    end_date: datetime
    kind: ShiftKind


def date_range(start_date: datetime, end_date: datetime, include_end_date: bool = True) -> Iterator[datetime]:
    """
    Creates an iterator that yields all dates between two given dates
    :param start_date:
    :param end_date:
    :param include_end_date
    :return:
    """
    for n in range(int((end_date - start_date).days) + int(include_end_date)):
        yield start_date + timedelta(n)


def create_shifts(start_date: datetime, end_date: datetime) -> List[Shift]:
    """
    Creates a list of all shifts between two dates (considers weekdays, weekends and holidays)
    :param start_date:
    :param end_date:
    :return:
    """
    shifts = []

    for date in date_range(start_date, end_date):
        if date.weekday() == THURSDAY:
            shifts.append(Shift(
                start_date=date,
                end_date=date + timedelta(days=3),
                kind=ShiftKind.WEEKEND
            ))
            continue

        if date.weekday() in WEEKEND:
            continue

        shifts.append(Shift(
            start_date=date,
            end_date=date + timedelta(days=1),
            kind=ShiftKind.WEEKDAY
        ))

    return shifts


def distribute_shifts_evenly(
        model: CpModel,
        shifts: Dict[Tuple[Monkey, Shift], IntVar],
        all_shifts: List[Shift],
        monkeys: List[Monkey],
        monkey_balance: Dict[Monkey, int],
):
    """
    Try to distribute day shifts evenly, so that each monkey works
    min_shifts_per_monkey shifts. If this is not possible, because the total
    number of shifts is not divisible by the number of nurses, some monkeys will
    be assigned one more shift.
    :param model:
    :param shifts:
    :param all_shifts:
    :param monkeys:
    :param monkey_balance:
    :return:
    """

    num_shifts = len(all_shifts)
    num_monkeys = len(monkeys)
    min_shifts_per_monkey = num_shifts // num_monkeys
    if num_shifts % num_monkeys == 0:
        max_shifts_per_monkey = min_shifts_per_monkey
    else:
        max_shifts_per_monkey = min_shifts_per_monkey + 1

    for monkey in monkeys:
        # TODO: what if monkey balance is too low to pass min_shifts_per_monkey
        num_shifts_worked = monkey_balance.get(monkey, 0)

        for index, shift in enumerate(all_shifts):
            num_shifts_worked += shifts[monkey, shift]

        model.Add(min_shifts_per_monkey <= num_shifts_worked)
        model.Add(num_shifts_worked <= max_shifts_per_monkey)


class ShiftKindInput:
    def __init__(self, shift_kind: ShiftKind, all_shifts: List[Shift], monkeys: List[Monkey],
                 balance: Dict[Monkey, int] = None):
        self.shift_kind = shift_kind
        self.shifts = list(filter(lambda shift: shift.kind is self.shift_kind, all_shifts))
        balance = balance or {}
        self.balance = {monkey: balance.get(monkey, 0) for monkey in monkeys}

    def get_stats(self) -> Tuple[int, int, Dict[Monkey, int]]:
        min_shifts = min(self.balance.values())
        max_shifts = max(self.balance.values())
        normalized_balance = {key: value - min_shifts for key, value in self.balance.items()}

        return min_shifts, max_shifts, normalized_balance

    def print_stats(self):
        min_shifts, max_shifts, normalized_balance = self.get_stats()

        rows = [
            self.shift_kind.value,
            f'  - min: {min_shifts}',
            f'  - max: {max_shifts}',
            f'  - balance: {normalized_balance}'
        ]
        print('\n'.join(rows))


def date_to_shifts_by_range(shifts: List[Shift], start_date: datetime, end_date: datetime = None) -> List[Shift]:
    """
    Filters a list of shifts to only the ones that are between two given days
    :param shifts:
    :param start_date:
    :param end_date:
    :return:
    """
    end_date = end_date or start_date
    return list(filter(lambda shift: start_date <= shift.start_date and shift.end_date <= end_date, shifts))


def main():
    all_shifts = create_shifts(start_date=datetime(2022, 1, 2), end_date=(datetime(2022, 2, 28)))
    date_to_shift = {shift.start_date: shift for shift in all_shifts}

    monkeys: List[Monkey] = [
        'one',
        'two',
        'three',
        'four',
        'five',
        'six',
        'seven',
        'eight',
        'nine'
    ]

    unavailability: Dict[Monkey, List[Shift]] = {
        'one': date_to_shifts_by_range(
            shifts=all_shifts,
            start_date=datetime(2022, 1, 1),
            end_date=datetime(2022, 1, 14)
        ),
    }

    soft_unavailability: Dict[Monkey, List[Shift]] = {
        'two': date_to_shifts_by_range(
            shifts=all_shifts,
            start_date=datetime(2022, 1, 1)
        )
    }

    preference: Dict[Monkey, List[Shift]] = {
        'three': [
            date_to_shift[datetime(2022, 1, 2)]
        ]
    }

    soft_preference: Dict[Monkey, List[Shift]] = {
        'four': date_to_shifts_by_range(
            shifts=all_shifts,
            start_date=datetime(2022, 2, 15),
            end_date=datetime(2022, 2, 16)
        )
    }

    shift_kind_to_input = {
        ShiftKind.WEEKDAY: ShiftKindInput(
            ShiftKind.WEEKDAY,
            all_shifts=all_shifts,
            monkeys=monkeys,
        ),
        ShiftKind.WEEKEND: ShiftKindInput(
            ShiftKind.WEEKEND,
            all_shifts=all_shifts,
            monkeys=monkeys,
            balance={'eight': -1, 'nine': -1}
        ),
        ShiftKind.HOLIDAY: ShiftKindInput(
            ShiftKind.HOLIDAY,
            all_shifts=all_shifts,
            monkeys=monkeys,
        ),
    }

    model = CpModel()

    # Set up the shifts dataset. This is a dict that resolves a combination of a monkey and a shift-slot to the boolean
    # variable that describes the possibility that the monkey will be assigned to that shift.
    shifts: Dict[Tuple[Monkey, Shift], IntVar] = {}

    for monkey in monkeys:
        for shift in all_shifts:
            shifts[monkey, shift] = model.NewBoolVar(f'shift_{monkey}_{shift}')

    # Each shift will be assigned to exactly one monkey.
    for shift in all_shifts:
        model.Add(sum(shifts[monkey, shift] for monkey in monkeys) == 1)

    # Availability / Preference

    # unavailability
    for monkey, monkey_shifts in unavailability.items():
        for shift in monkey_shifts:
            model.Add(shifts[monkey, shift] == 0)

    # soft-unavailability
    num_unavailable = 0
    for monkey, monkey_shifts in soft_unavailability.items():
        for shift in monkey_shifts:
            num_unavailable += shifts[monkey, shift]

    model.Minimize(num_unavailable)

    # preference
    for monkey, monkey_shifts in preference.items():
        for shift in monkey_shifts:
            model.Add(shifts[monkey, shift] == 1)

    # soft-preference
    num_prefer = 0
    for monkey, monkey_shifts in soft_preference.items():
        for shift in monkey_shifts:
            num_prefer += shifts[monkey, shift]

    model.Maximize(num_prefer)

    # Distribute shifts evenly for each shift kind
    for shift_kind, shift_kind_input in shift_kind_to_input.items():
        distribute_shifts_evenly(
            model=model,
            shifts=shifts,
            all_shifts=shift_kind_input.shifts,
            monkeys=monkeys,
            monkey_balance=shift_kind_input.balance
        )

    # There will be at least hard_min rest-days between each shift
    hard_min = 5
    for monkey in monkeys:

        works = [shifts[monkey, shift] for shift in all_shifts]

        for i, work in enumerate(works):
            actual_min = min(hard_min, len(works) - 1 - i)

            model.AddBoolAnd([
                sequence_work.Not()
                for sequence_work in works[i + 1: i + actual_min]
            ]).OnlyEnforceIf(
                work
            )

    solver = CpSolver()
    status = solver.Solve(model)

    if status != cp_model.OPTIMAL:
        print('No optimal solution found !')
    else:
        print('Solution:')
        for shift in all_shifts:
            for monkey in monkeys:
                if not solver.Value(shifts[monkey, shift]) == 1:
                    continue

                shift_kind_to_input[shift.kind].balance[monkey] += 1

                for date in date_range(shift.start_date, shift.end_date, include_end_date=False):
                    print(f'{date.strftime("%b-%d")} | {date.strftime("%a")} | {monkey}')

        print('\n\nStatistics:')

        for shift_kind_input in shift_kind_to_input.values():
            shift_kind_input.print_stats()

        print('\nSolution Statistics:')
        print('  - conflicts: %i' % solver.NumConflicts())
        print('  - branches : %i' % solver.NumBranches())
        print('  - wall time: %f s' % solver.WallTime())

        return


if __name__ == '__main__':
    main()
