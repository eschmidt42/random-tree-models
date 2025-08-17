from rich import print as rprint
from rich.tree import Tree

from random_tree_models.decisiontree.estimators import DecisionTreeTemplate
from random_tree_models.decisiontree.node import Node


def walk_tree(
    decision_tree: Node,
    tree: Tree,
    parent: Node | None = None,
    is_left: bool | None = None,
):
    arrow = (
        ""
        if parent is None
        else f"[magenta](< {parent.threshold:.3f})[/magenta]"
        if is_left
        else f"[magenta](>= {parent.threshold:.3f})[/magenta]"
    )

    if decision_tree.is_leaf:  # base cases
        branch = tree.add(
            f"{arrow} 🍁 # obs: [cyan]{decision_tree.n_obs}[/cyan], value: [green]{decision_tree.prediction:.3f}[/green], leaf reason '{decision_tree.reason}'"
        )
        return None
    else:
        branch = tree.add(
            f"{arrow} col idx: {decision_tree.array_column}, threshold: [magenta]{decision_tree.threshold:.3f}[/magenta]"
        )

        if decision_tree.left is not None:  # go left
            walk_tree(decision_tree.left, branch, decision_tree, True)

        if decision_tree.right is not None:  # go right
            walk_tree(decision_tree.right, branch, decision_tree, False)


def show_tree(decision_tree: DecisionTreeTemplate):
    tree = Tree(f"Represenation of 🌲 ({decision_tree})")
    walk_tree(decision_tree.tree_, tree)
    rprint(tree)
