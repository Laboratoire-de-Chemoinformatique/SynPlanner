from collections import defaultdict
from functools import partial
from math import hypot
from uuid import uuid4

from CGRtools.algorithms.depict import (
    Depict,
    DepictCGR,
    DepictMolecule,
    _render_charge,
    rotate_vector,
)
from CGRtools.containers import CGRContainer, MoleculeContainer, ReactionContainer


class WideBondDepictCGR(DepictCGR):
    """
    Like DepictCGR, but all DynamicBonds
    are drawn 2.5× wider than the standard bond width.
    """

    __slots__ = ()

    def _render_bonds(self):
        """
        Renders the bonds of the CGR as SVG lines, with DynamicBonds drawn wider.

        This method overrides the base `_render_bonds` to apply a wider stroke
        to DynamicBonds, highlighting changes in bond order during a reaction.
        It iterates through all bonds, calculates their positions based on
        2D coordinates, and generates SVG `<line>` elements with appropriate
        styles (color, width, dash array) based on the bond's original (`order`)
        and primary (`p_order`) states. Aromatic bonds are handled separately
        using a helper method.

        Returns:
            list: A list of strings, where each string is an SVG element
                  representing a bond.
        """
        plane = self._plane
        config = self._render_config

        # get the normal width (default 1.0) and compute a 4× wide stroke
        normal_width = config.get("bond_width", 0.02)
        wide_width = normal_width * 2.5

        broken = config["broken_color"]
        formed = config["formed_color"]
        dash1, dash2 = config["dashes"]
        double_space = config["double_space"]
        triple_space = config["triple_space"]

        svg = []
        ar_bond_colors = defaultdict(dict)

        for n, m, bond in self.bonds():
            order, p_order = bond.order, bond.p_order
            nx, ny = plane[n]
            mx, my = plane[m]
            # invert Y for SVG
            ny, my = -ny, -my
            rv = partial(rotate_vector, 0, x2=mx - nx, y2=ny - my)
            if order == 1:
                if p_order == 1:
                    svg.append(
                        f'      <line x1="{nx:.2f}" y1="{ny:.2f}" x2="{mx:.2f}" y2="{my:.2f}"/>'
                    )
                elif p_order == 4:
                    ar_bond_colors[n][m] = ar_bond_colors[m][n] = formed
                    svg.append(
                        f'      <line x1="{nx:.2f}" y1="{ny:.2f}" x2="{mx:.2f}" y2="{my:.2f}"/>'
                    )
                elif p_order == 2:
                    dx, dy = rv(double_space)
                    svg.append(
                        f'      <line x1="{nx + dx:.2f}" y1="{ny - dy:.2f}" '
                        f'x2="{mx + dx:.2f}" y2="{my - dy:.2f}"/>'
                    )
                    svg.append(
                        f'      <line x1="{nx - dx:.2f}" y1="{ny + dy:.2f}" '
                        f'x2="{mx - dx:.2f}" y2="{my + dy:.2f}" stroke="{formed}" stroke-width="{wide_width:.2f}"/>'
                    )
                elif p_order == 3:
                    dx, dy = rv(triple_space)
                    svg.append(
                        f'      <line x1="{nx:.2f}" y1="{ny:.2f}" x2="{mx:.2f}" y2="{my:.2f}"'
                        f' stroke="{formed}" stroke-width="{wide_width:.2f}"/>'
                    )
                    svg.append(
                        f'      <line x1="{nx + dx:.2f}" y1="{ny - dy:.2f}" '
                        f'x2="{mx + dx:.2f}" y2="{my - dy:.2f}" stroke-width="{wide_width:.2f}"/>'
                    )
                    svg.append(
                        f'      <line x1="{nx - dx:.2f}" y1="{ny + dy:.2f}" '
                        f'x2="{mx - dx:.2f}" y2="{my + dy:.2f}" stroke="{formed}" stroke-width="{wide_width:.2f}"/>'
                    )
                elif p_order is None:
                    svg.append(
                        f'      <line x1="{nx:.2f}" y1="{ny:.2f}" x2="{mx:.2f}" y2="{my:.2f}"'
                        f' stroke="{broken}" stroke-width="{wide_width:.2f}"/>'
                    )
                else:
                    dx, dy = rv(double_space)
                    svg.append(
                        f'      <line x1="{nx + dx:.2f}" y1="{ny - dy:.2f}" x2="{mx + dx:.2f}"'
                        f' y2="{my - dy:.2f}" stroke-dasharray="{dash1:.2f} {dash2:.2f}" stroke="{formed}" stroke-width="{wide_width:.2f}"/>'
                    )
                    svg.append(
                        f'      <line x1="{nx - dx:.2f}" y1="{ny + dy:.2f}" '
                        f'x2="{mx - dx:.2f}" y2="{my + dy:.2f}" stroke="{broken}" stroke-width="{wide_width:.2f}"/>'
                    )
            elif order == 4:
                if p_order == 4:
                    svg.append(
                        f'      <line x1="{nx:.2f}" y1="{ny:.2f}" x2="{mx:.2f}" y2="{my:.2f}"/>'
                    )
                elif p_order == 1:
                    ar_bond_colors[n][m] = ar_bond_colors[m][n] = broken
                    svg.append(
                        f'      <line x1="{nx:.2f}" y1="{ny:.2f}" x2="{mx:.2f}" y2="{my:.2f}"/>'
                    )
                elif p_order == 2:
                    ar_bond_colors[n][m] = ar_bond_colors[m][n] = broken
                    dx, dy = rv(double_space)
                    svg.append(
                        f'      <line x1="{nx + dx:.2f}" y1="{ny - dy:.2f}" '
                        f'x2="{mx + dx:.2f}" y2="{my - dy:.2f}"/>'
                    )
                    svg.append(
                        f'      <line x1="{nx - dx:.2f}" y1="{ny + dy:.2f}" '
                        f'x2="{mx - dx:.2f}" y2="{my + dy:.2f}" stroke="{formed}" stroke-width="{wide_width:.2f}"/>'
                    )
                elif p_order == 3:
                    ar_bond_colors[n][m] = ar_bond_colors[m][n] = broken
                    dx, dy = rv(triple_space)
                    svg.append(
                        f'      <line x1="{nx + dx:.2f}" y1="{ny - dy:.2f}" '
                        f'x2="{mx + dx:.2f}" y2="{my - dy:.2f}"  stroke="{formed}" stroke-width="{wide_width:.2f}"/>'
                    )
                    svg.append(
                        f'      <line x1="{nx:.2f}" y1="{ny:.2f}" x2="{mx:.2f}" y2="{my:.2f}"/>'
                    )
                    svg.append(
                        f'      <line x1="{nx - dx:.2f}" y1="{ny + dy:.2f}" '
                        f'x2="{mx - dx:.2f}" y2="{my + dy:.2f}" stroke="{formed}" stroke-width="{wide_width:.2f}"/>'
                    )
                elif p_order is None:
                    ar_bond_colors[n][m] = ar_bond_colors[m][n] = broken
                    svg.append(
                        f'      <line x1="{nx:.2f}" y1="{ny:.2f}" x2="{mx:.2f}" y2="{my:.2f}"'
                        f' stroke="{broken}" stroke-width="{wide_width:.2f}"/>'
                    )
                else:
                    ar_bond_colors[n][m] = ar_bond_colors[m][n] = None
                    svg.append(
                        f'      <line x1="{nx:.2f}" y1="{ny:.2f}" x2="{mx:.2f}" y2="{my:.2f}"'
                        f' stroke="{broken}" stroke-width="{wide_width:.2f}"/>'
                    )
            elif order == 2:
                if p_order == 2:
                    dx, dy = rv(double_space)
                    svg.append(
                        f'      <line x1="{nx + dx:.2f}" y1="{ny - dy:.2f}" '
                        f'x2="{mx + dx:.2f}" y2="{my - dy:.2f}"/>'
                    )
                    svg.append(
                        f'      <line x1="{nx - dx:.2f}" y1="{ny + dy:.2f}" '
                        f'x2="{mx - dx:.2f}" y2="{my + dy:.2f}"/>'
                    )
                elif p_order == 1:
                    dx, dy = rv(double_space)
                    svg.append(
                        f'      <line x1="{nx + dx:.2f}" y1="{ny - dy:.2f}" '
                        f'x2="{mx + dx:.2f}" y2="{my - dy:.2f}"/>'
                    )
                    svg.append(
                        f'      <line x1="{nx - dx:.2f}" y1="{ny + dy:.2f}" '
                        f'x2="{mx - dx:.2f}" y2="{my + dy:.2f}" stroke="{broken}" stroke-width="{wide_width:.2f}"/>'
                    )
                elif p_order == 4:
                    ar_bond_colors[n][m] = ar_bond_colors[m][n] = formed
                    dx, dy = rv(double_space)
                    svg.append(
                        f'      <line x1="{nx + dx:.2f}" y1="{ny - dy:.2f}" '
                        f'x2="{mx + dx:.2f}" y2="{my - dy:.2f}"/>'
                    )
                    svg.append(
                        f'      <line x1="{nx - dx:.2f}" y1="{ny + dy:.2f}" '
                        f'x2="{mx - dx:.2f}" y2="{my + dy:.2f}" stroke="{broken}" stroke-width="{wide_width:.2f}"/>'
                    )
                elif p_order == 3:
                    dx, dy = rv(triple_space)
                    svg.append(
                        f'      <line x1="{nx + dx:.2f}" y1="{ny - dy:.2f}" '
                        f'x2="{mx + dx:.2f}" y2="{my - dy:.2f}"/>'
                    )
                    svg.append(
                        f'      <line x1="{nx:.2f}" y1="{ny:.2f}" x2="{mx:.2f}" y2="{my:.2f}"/>'
                    )
                    svg.append(
                        f'      <line x1="{nx - dx:.2f}" y1="{ny + dy:.2f}" '
                        f'x2="{mx - dx:.2f}" y2="{my + dy:.2f}" stroke="{formed} stroke-width="{wide_width:.2f}""/>'
                    )
                elif p_order is None:
                    dx, dy = rv(double_space)
                    svg.append(
                        f'      <line x1="{nx + dx:.2f}" y1="{ny - dy:.2f}" '
                        f'x2="{mx + dx:.2f}" y2="{my - dy:.2f}" stroke="{broken}" stroke-width="{wide_width:.2f}"/>'
                    )
                    svg.append(
                        f'      <line x1="{nx - dx:.2f}" y1="{ny + dy:.2f}" '
                        f'x2="{mx - dx:.2f}" y2="{my + dy:.2f}" stroke="{broken}" stroke-width="{wide_width:.2f}"/>'
                    )
                else:
                    dx, dy = rv(triple_space)
                    svg.append(
                        f'      <line x1="{nx + dx:.2f}" y1="{ny - dy:.2f}" x2="{mx + dx:.2f}"'
                        f' y2="{my - dy:.2f}" stroke-dasharray="{dash1:.2f} {dash2:.2f}" stroke="{formed}" stroke-width="{wide_width:.2f}"/>'
                    )
                    svg.append(
                        f'      <line x1="{nx:.2f}" y1="{ny:.2f}" x2="{mx:.2f}" y2="{my:.2f}"'
                        f' stroke="{broken}" stroke-width="{wide_width:.2f}"/>'
                    )
                    svg.append(
                        f'      <line x1="{nx - dx:.2f}" y1="{ny + dy:.2f}" '
                        f'x2="{mx - dx:.2f}" y2="{my + dy:.2f}" stroke="{broken}" stroke-width="{wide_width:.2f}"/>'
                    )
            elif order == 3:
                if p_order == 3:
                    dx, dy = rv(triple_space)
                    svg.append(
                        f'      <line x1="{nx + dx:.2f}" y1="{ny - dy:.2f}" '
                        f'x2="{mx + dx:.2f}" y2="{my - dy:.2f}"/>'
                    )
                    svg.append(
                        f'      <line x1="{nx:.2f}" y1="{ny:.2f}" x2="{mx:.2f}" y2="{my:.2f}"/>'
                    )
                    svg.append(
                        f'      <line x1="{nx - dx:.2f}" y1="{ny + dy:.2f}" '
                        f'x2="{mx - dx:.2f}" y2="{my + dy:.2f}"/>'
                    )
                elif p_order == 1:
                    dx, dy = rv(triple_space)
                    svg.append(
                        f'      <line x1="{nx + dx:.2f}" y1="{ny - dy:.2f}" '
                        f'x2="{mx + dx:.2f}" y2="{my - dy:.2f}"/>'
                    )
                    svg.append(
                        f'      <line x1="{nx:.2f}" y1="{ny:.2f}" x2="{mx:.2f}" y2="{my:.2f}"'
                        f' stroke="{broken}" stroke-width="{wide_width:.2f}"/>'
                    )
                    svg.append(
                        f'      <line x1="{nx - dx:.2f}" y1="{ny + dy:.2f}" '
                        f'x2="{mx - dx:.2f}" y2="{my + dy:.2f}" '
                        f'stroke="{broken}" stroke-width="{wide_width:.2f}"/>'
                    )
                elif p_order == 4:
                    ar_bond_colors[n][m] = ar_bond_colors[m][n] = formed
                    dx, dy = rv(triple_space)
                    svg.append(
                        f'      <line x1="{nx + dx:.2f}" y1="{ny - dy:.2f}" x2="{mx + dx:.2f}" '
                        f'y2="{my - dy:.2f}" stroke="{broken}" stroke-width="{wide_width:.2f}"/>'
                    )
                    svg.append(
                        f'      <line x1="{nx:.2f}" y1="{ny:.2f}" x2="{mx:.2f}" y2="{my:.2f}"/>'
                    )
                    svg.append(
                        f'      <line x1="{nx - dx:.2f}" y1="{ny + dy:.2f}" x2="{mx - dx:.2f}" '
                        f'y2="{my + dy:.2f}" stroke="{broken}" stroke-width="{wide_width:.2f}"/>'
                    )
                elif p_order == 2:
                    dx, dy = rv(triple_space)
                    svg.append(
                        f'      <line x1="{nx + dx:.2f}" y1="{ny - dy:.2f}" '
                        f'x2="{mx + dx:.2f}" y2="{my - dy:.2f}"/>'
                    )
                    svg.append(
                        f'      <line x1="{nx:.2f}" y1="{ny:.2f}" x2="{mx:.2f}" y2="{my:.2f}"/>'
                    )
                    svg.append(
                        f'      <line x1="{nx - dx:.2f}" y1="{ny + dy:.2f}" '
                        f'x2="{mx - dx:.2f}" y2="{my + dy:.2f}" stroke="{broken}" stroke-width="{wide_width:.2f}"/>'
                    )
                elif p_order is None:
                    dx, dy = rv(triple_space)
                    svg.append(
                        f'      <line x1="{nx + dx:.2f}" y1="{ny - dy:.2f}" '
                        f'x2="{mx + dx:.2f}" y2="{my - dy:.2f}" stroke="{broken}" stroke-width="{wide_width:.2f}"/>'
                    )
                    svg.append(
                        f'      <line x1="{nx:.2f}" y1="{ny:.2f}" '
                        f'x2="{mx:.2f}" y2="{my:.2f}" stroke="{broken}" stroke-width="{wide_width:.2f}"/>'
                    )
                    svg.append(
                        f'      <line x1="{nx - dx:.2f}" y1="{ny + dy:.2f}" '
                        f'x2="{mx - dx:.2f}" y2="{my + dy:.2f}" stroke="{broken}" stroke-width="{wide_width:.2f}"/>'
                    )
                else:
                    dx, dy = rv(double_space)
                    dx3 = 3 * dx
                    dy3 = 3 * dy
                    svg.append(
                        f'      <line x1="{nx + dx3:.2f}" y1="{ny - dy3:.2f}" x2="{mx + dx3:.2f}" '
                        f'y2="{my - dy3:.2f}" stroke-dasharray="{dash1:.2f} {dash2:.2f}" stroke="{formed}" stroke-width="{wide_width:.2f}"/>'
                    )
                    svg.append(
                        f'      <line x1="{nx + dx:.2f}" y1="{ny - dy:.2f}" '
                        f'x2="{mx + dx:.2f}" y2="{my - dy:.2f}" stroke="{broken}" stroke-width="{wide_width:.2f}"/>'
                    )
                    svg.append(
                        f'      <line x1="{nx - dx:.2f}" y1="{ny + dy:.2f}" '
                        f'x2="{mx - dx:.2f}" y2="{my + dy:.2f}" stroke="{broken}" stroke-width="{wide_width:.2f}"/>'
                    )
                    svg.append(
                        f'      <line x1="{nx - dx3:.2f}" y1="{ny + dy3:.2f}" x2="{mx - dx3:.2f}" '
                        f'y2="{my + dy3:.2f}" stroke="{broken}" stroke-width="{wide_width:.2f}"/>'
                    )
            elif order is None:
                if p_order == 1:
                    svg.append(
                        f'      <line x1="{nx:.2f}" y1="{ny:.2f}" x2="{mx:.2f}" y2="{my:.2f}"'
                        f' stroke="{formed}" stroke-width="{wide_width:.2f}"/>'
                    )
                elif p_order == 4:
                    ar_bond_colors[n][m] = ar_bond_colors[m][n] = formed
                    svg.append(
                        f'      <line x1="{nx:.2f}" y1="{ny:.2f}" x2="{mx:.2f}" y2="{my:.2f}"'
                        f' stroke="{formed}" stroke-width="{wide_width:.2f}"/>'
                    )
                elif p_order == 2:
                    dx, dy = rv(double_space)
                    # dx = dx // 1.4
                    # dy = dy // 1.4
                    svg.append(
                        f'      <line x1="{nx + dx:.2f}" y1="{ny - dy:.2f}" x2="{mx + dx:.2f}" '
                        f'y2="{my - dy:.2f}" stroke="{formed}" stroke-width="{wide_width:.2f}"/>'
                    )
                    svg.append(
                        f'      <line x1="{nx - dx:.2f}" y1="{ny + dy:.2f}" x2="{mx - dx:.2f}" '
                        f'y2="{my + dy:.2f}" stroke="{formed}" stroke-width="{wide_width:.2f}"/>'
                    )
                elif p_order == 3:
                    dx, dy = rv(triple_space)
                    svg.append(
                        f'      <line x1="{nx + dx:.2f}" y1="{ny - dy:.2f}" '
                        f'x2="{mx + dx:.2f}" y2="{my - dy:.2f}" stroke="{formed}" stroke-width="{wide_width:.2f}"/>'
                    )
                    svg.append(
                        f'      <line x1="{nx:.2f}" y1="{ny:.2f}" x2="{mx:.2f}" y2="{my:.2f}"'
                        f' stroke="{formed}" stroke-width="{wide_width:.2f}"/>'
                    )
                    svg.append(
                        f'      <line x1="{nx - dx:.2f}" y1="{ny + dy:.2f}" '
                        f'x2="{mx - dx:.2f}" y2="{my + dy:.2f}" stroke="{formed}" stroke-width="{wide_width:.2f}"/>'
                    )
                else:
                    svg.append(
                        f'      <line x1="{nx:.2f}" y1="{ny:.2f}" x2="{mx:.2f}" y2="{my:.2f}" '
                        f'stroke-dasharray="{dash1:.2f} {dash2:.2f}" stroke="{formed}" stroke-width="{wide_width:.2f}"/>'
                    )
            else:
                if p_order == 8:
                    svg.append(
                        f'        <line x1="{nx:.2f}" y1="{ny:.2f}" x2="{mx:.2f}" y2="{my:.2f}" '
                        f'stroke-dasharray="{dash1:.2f} {dash2:.2f}"/>'
                    )
                elif p_order == 1:
                    dx, dy = rv(double_space)
                    svg.append(
                        f'      <line x1="{nx + dx:.2f}" y1="{ny - dy:.2f}" x2="{mx + dx:.2f}"'
                        f' y2="{my - dy:.2f}" stroke-dasharray="{dash1:.2f} {dash2:.2f}" stroke="{broken}" stroke-width="{wide_width:.2f}"/>'
                    )
                    svg.append(
                        f'      <line x1="{nx - dx:.2f}" y1="{ny + dy:.2f}" '
                        f'x2="{mx - dx:.2f}" y2="{my + dy:.2f}" stroke="{formed}" stroke-width="{wide_width:.2f}"/>'
                    )
                elif p_order == 4:
                    ar_bond_colors[n][m] = ar_bond_colors[m][n] = None
                    svg.append(
                        f'      <line x1="{nx:.2f}" y1="{ny:.2f}" x2="{mx:.2f}" y2="{my:.2f}"'
                        f' stroke="{formed}" stroke-width="{wide_width:.2f}"/>'
                    )
                elif p_order == 2:
                    dx, dy = rv(triple_space)
                    svg.append(
                        f'      <line x1="{nx + dx:.2f}" y1="{ny - dy:.2f}" x2="{mx + dx:.2f}"'
                        f' y2="{my - dy:.2f}" stroke-dasharray="{dash1:.2f} {dash2:.2f}" stroke="{broken}" stroke-width="{wide_width:.2f}"/>'
                    )
                    svg.append(
                        f'      <line x1="{nx:.2f}" y1="{ny:.2f}" x2="{mx:.2f}" y2="{my:.2f}"'
                        f' stroke="{formed}" stroke-width="{wide_width:.2f}"/>'
                    )
                    svg.append(
                        f'      <line x1="{nx - dx:.2f}" y1="{ny + dy:.2f}" '
                        f'x2="{mx - dx:.2f}" y2="{my + dy:.2f}" stroke="{formed}" stroke-width="{wide_width:.2f}"/>'
                    )
                elif p_order == 3:
                    dx, dy = rv(double_space)
                    dx3 = 3 * dx
                    dy3 = 3 * dy
                    svg.append(
                        f'      <line x1="{nx + dx3:.2f}" y1="{ny - dy3:.2f}" x2="{mx + dx3:.2f}" '
                        f'y2="{my - dy3:.2f}" stroke-dasharray="{dash1:.2f} {dash2:.2f}" stroke="{broken}" stroke-width="{wide_width:.2f}"/>'
                    )
                    svg.append(
                        f'      <line x1="{nx + dx:.2f}" y1="{ny - dy:.2f}" '
                        f'x2="{mx + dx:.2f}" y2="{my - dy:.2f}" stroke="{formed}" stroke-width="{wide_width:.2f}"/>'
                    )
                    svg.append(
                        f'      <line x1="{nx - dx:.2f}" y1="{ny + dy:.2f}" '
                        f'x2="{mx - dx:.2f}" y2="{my + dy:.2f}" stroke="{formed}" stroke-width="{wide_width:.2f}"/>'
                    )
                    svg.append(
                        f'      <line x1="{nx - dx3:.2f}" y1="{ny + dy3:.2f}" '
                        f'x2="{mx - dx3:.2f}" y2="{my + dy3:.2f}" stroke="{formed}" stroke-width="{wide_width:.2f}"/>'
                    )
                else:
                    svg.append(
                        f'      <line x1="{nx:.2f}" y1="{ny:.2f}" x2="{mx:.2f}" y2="{my:.2f}" '
                        f'stroke-dasharray="{dash1:.2f} {dash2:.2f}" stroke="{broken}" stroke-width="{wide_width:.2f}"/>'
                    )

        # aromatic rings - unchanged
        for ring in self.aromatic_rings:
            cx = sum(plane[x][0] for x in ring) / len(ring)
            cy = sum(plane[x][1] for x in ring) / len(ring)

            for n, m in zip(ring, ring[1:]):
                nx, ny = plane[n]
                mx, my = plane[m]
                aromatic = self.__render_aromatic_bond(
                    nx, ny, mx, my, cx, cy, ar_bond_colors[n].get(m)
                )
                if aromatic:
                    svg.append(aromatic)

            n, m = ring[-1], ring[0]
            nx, ny = plane[n]
            mx, my = plane[m]
            aromatic = self.__render_aromatic_bond(
                nx, ny, mx, my, cx, cy, ar_bond_colors[n].get(m)
            )
            if aromatic:
                svg.append(aromatic)
        return svg

    def __render_aromatic_bond(self, n_x, n_y, m_x, m_y, c_x, c_y, color):
        config = self._render_config

        dash1, dash2 = config["dashes"]
        dash3, dash4 = config["aromatic_dashes"]
        aromatic_space = config["cgr_aromatic_space"]

        normal_width = config.get("bond_width", 0.02)
        wide_width = normal_width * 2

        # n aligned xy
        mn_x, mn_y, cn_x, cn_y = m_x - n_x, m_y - n_y, c_x - n_x, c_y - n_y

        # nm reoriented xy
        mr_x, mr_y = hypot(mn_x, mn_y), 0
        cr_x, cr_y = rotate_vector(cn_x, cn_y, mn_x, -mn_y)

        if cr_y and aromatic_space / cr_y < 0.65:
            if cr_y > 0:
                r_y = aromatic_space
            else:
                r_y = -aromatic_space
                cr_y = -cr_y

            ar_x = aromatic_space * cr_x / cr_y
            br_x = mr_x - aromatic_space * (mr_x - cr_x) / cr_y

            # backward reorienting
            an_x, an_y = rotate_vector(ar_x, r_y, mn_x, mn_y)
            bn_x, bn_y = rotate_vector(br_x, r_y, mn_x, mn_y)

            if color:
                # print('color')
                return (
                    f'      <line x1="{an_x + n_x:.2f}" y1="{-an_y - n_y:.2f}" x2="{bn_x + n_x:.2f}" '
                    f'y2="{-bn_y - n_y:.2f}" stroke-dasharray="{dash3:.2f} {dash4:.2f}" stroke="{color}" stroke-width="{wide_width:.2f}"/>'
                )
            elif color is None:
                dash3, dash4 = dash1, dash2
            return (
                f'      <line x1="{an_x + n_x:.2f}" y1="{-an_y - n_y:.2f}"'
                f' x2="{bn_x + n_x:.2f}" y2="{-bn_y - n_y:.2f}" stroke-dasharray="{dash3:.2f} {dash4:.2f}"/>'
            )


def cgr_display(cgr: CGRContainer) -> str:
    """
    Generates an SVG string for displaying a CGR with wider DynamicBonds.

    This function temporarily modifies the rendering methods of the
    `CGRContainer` class to use the bond rendering logic from
    `WideBondDepictCGR`, which draws DynamicBonds with a wider stroke.
    It cleans the 2D coordinates of the input CGR and then calls its
    `depict()` method to generate the SVG string using the modified
    rendering behavior.

    Args:
        cgr (CGRContainer): The CGRContainer object to be depicted.

    Returns:
        str: An SVG string representing the depiction of the CGR
             with wider DynamicBonds.
    """
    CGRContainer._CGRContainer__render_aromatic_bond = (
        WideBondDepictCGR._WideBondDepictCGR__render_aromatic_bond
    )
    CGRContainer._render_bonds = WideBondDepictCGR._render_bonds
    CGRContainer._WideBondDepictCGR__render_aromatic_bond = (
        WideBondDepictCGR._WideBondDepictCGR__render_aromatic_bond
    )
    cgr.clean2d()
    return cgr.depict()


class CustomDepictMolecule(DepictMolecule):
    """
    Custom molecule depiction class that uses atom.symbol for rendering.
    """

    def _render_atoms(self):
        bonds = self._bonds
        plane = self._plane
        charges = self._charges
        radicals = self._radicals
        hydrogens = self._hydrogens
        config = self._render_config

        carbon = config["carbon"]
        mapping = config["mapping"]
        span_size = config["span_size"]
        font_size = config["font_size"]
        monochrome = config["monochrome"]
        other_size = config["other_size"]
        atoms_colors = config["atoms_colors"]
        mapping_font = config["mapping_size"]
        dx_m, dy_m = config["dx_m"], config["dy_m"]
        dx_ci, dy_ci = config["dx_ci"], config["dy_ci"]
        symbols_font_style = config["symbols_font_style"]

        # for cumulenes
        try:
            # Check if _cumulenes method exists and handle potential errors
            cumulenes = {
                y
                for x in self._cumulenes(heteroatoms=True)
                if len(x) > 2
                for y in x[1:-1]
            }
        except AttributeError:
            cumulenes = set()  # Fallback if _cumulenes is not available or fails

        if monochrome:
            map_fill = other_fill = "black"
        else:
            map_fill = config["mapping_color"]
            other_fill = config["other_color"]

        svg = []
        maps = []
        others = []
        font2 = 0.2 * font_size
        font3 = 0.3 * font_size
        font4 = 0.4 * font_size
        font5 = 0.5 * font_size
        font6 = 0.6 * font_size
        font7 = 0.7 * font_size
        font15 = 0.15 * font_size
        font25 = 0.25 * font_size
        mask = defaultdict(list)
        for n, atom in self._atoms.items():
            x, y = plane[n]
            y = -y

            # --- KEY CHANGE HERE ---
            # Use atom.symbol if it exists, otherwise fallback to atomic_symbol
            try:
                symbol = atom.symbol
            except AttributeError:
                symbol = atom.atomic_symbol  # Fallback if .symbol doesn't exist
            # --- END KEY CHANGE ---

            if (
                not bonds.get(n)
                or symbol != "C"
                or carbon
                or atom.charge
                or atom.is_radical
                or atom.isotope
                or n in cumulenes
            ):  # Added bonds.get(n) check for single atoms
                # Calculate hydrogens if the attribute exists, otherwise default to 0
                try:
                    h = hydrogens[n]
                except (KeyError, AttributeError):
                    h = 0  # Default if _hydrogens is missing or key n is not present

                if h == 1:
                    h_str = "H"
                    span = ""
                elif h and h > 1:  # Check if h is not None and greater than 1
                    span = f'<tspan  dy="{config["span_dy"]:.2f}" font-size="{span_size:.2f}">{h}</tspan>'
                    h_str = "H"
                else:
                    h_str = ""
                    span = ""

                # Handle charges and radicals safely
                charge_val = charges.get(n, 0)
                is_radical = radicals.get(n, False)

                if charge_val:
                    t = f'{_render_charge.get(charge_val, "")}{"↑" if is_radical else ""}'  # Use .get for safety
                    if t:  # Only add if charge text is generated
                        others.append(
                            f'        <text x="{x:.2f}" y="{y:.2f}" dx="{dx_ci:.2f}" dy="-{dy_ci:.2f}">'
                            f"{t}</text>"
                        )
                        mask["other"].append(
                            f'           <text x="{x:.2f}" y="{y:.2f}" dx="{dx_ci:.2f}" dy="-{dy_ci:.2f}">'
                            f"{t}</text>"
                        )
                elif is_radical:
                    others.append(
                        f'        <text x="{x:.2f}" y="{y:.2f}" dx="{dx_ci:.2f}" dy="-{dy_ci:.2f}">↑</text>'
                    )
                    mask["other"].append(
                        f'            <text x="{x:.2f}" y="{y:.2f}" dx="{dx_ci:.2f}"'
                        f' dy="-{dy_ci:.2f}">↑</text>'
                    )

                # Handle isotope safely
                try:
                    iso = atom.isotope
                    if iso:
                        t = iso
                        others.append(
                            f'        <text x="{x:.2f}" y="{y:.2f}" dx="-{dx_ci:.2f}" dy="-{dy_ci:.2f}" '
                            f'text-anchor="end">{t}</text>'
                        )
                        mask["other"].append(
                            f'            <text x="{x:.2f}" y="{y:.2f}" dx="-{dx_ci:.2f}"'
                            f' dy="-{dy_ci:.2f}" text-anchor="end">{t}</text>'
                        )
                except AttributeError:
                    pass  # Atom might not have isotope attribute

                # Determine atom color based on atomic_number, default to black if monochrome or not found
                atom_color = "black"
                if not monochrome:
                    try:
                        an = atom.atomic_number
                        if 0 < an <= len(atoms_colors):
                            atom_color = atoms_colors[an - 1]
                        else:
                            atom_color = atoms_colors[
                                5
                            ]  # Default to Carbon color if out of range
                    except AttributeError:
                        atom_color = atoms_colors[
                            5
                        ]  # Default to Carbon color if no atomic_number

                svg.append(
                    f'      <g fill="{atom_color}" '
                    f'font-family="{symbols_font_style }">'
                )

                # Adjust dx based on symbol length for better centering
                if len(symbol) > 1:
                    dx = font7
                    dx_mm = dx_m + font5
                    if symbol[-1].lower() in (
                        "l",
                        "i",
                        "r",
                        "t",
                    ):  # Heuristic for narrow last letters
                        rx = font6
                        ax = font25
                    else:
                        rx = font7
                        ax = font15
                    mask["center"].append(
                        f'          <ellipse cx="{x - ax:.2f}" cy="{y:.2f}" rx="{rx}" ry="{font4}"/>'
                    )
                else:
                    if symbol == "I":  # Special case for 'I'
                        dx = font15
                        dx_mm = dx_m
                    else:  # Single character
                        dx = font4
                        dx_mm = dx_m + font2
                    mask["center"].append(
                        f'          <circle cx="{x:.2f}" cy="{y:.2f}" r="{font4:.2f}"/>'
                    )

                svg.append(
                    f'        <text x="{x:.2f}" y="{y:.2f}" dx="-{dx:.2f}" dy="{font4:.2f}" '
                    f'font-size="{font_size:.2f}">{symbol}{h_str}{span}</text>'
                )
                mask["symbols"].append(
                    f'            <text x="{x:.2f}" y="{y:.2f}" dx="-{dx:.2f}" '
                    f'dy="{font4:.2f}">{symbol}{h_str}</text>'
                )
                if span:
                    mask["span"].append(
                        f'            <text x="{x:.2f}" y="{y:.2f}" dx="-{dx:.2f}" dy="{font4:.2f}">'
                        f"{symbol}{h_str}{span}</text>"
                    )
                svg.append("      </g>")

                if mapping:
                    maps.append(
                        f'        <text x="{x:.2f}" y="{y:.2f}" dx="-{dx_mm:.2f}" dy="{dy_m + font3:.2f}" '
                        f'text-anchor="end">{n}</text>'
                    )
                    mask["aam"].append(
                        f'            <text x="{x:.2f}" y="{y:.2f}" dx="-{dx_mm:.2f}" '
                        f'dy="{dy_m + font3:.2f}" text-anchor="end">{n}</text>'
                    )

            elif mapping:
                # Determine dx_mm for mapping based on symbol length even if atom itself isn't drawn
                if len(symbol) > 1:
                    dx_mm = dx_m + font5
                else:
                    dx_mm = dx_m + font2 if symbol != "I" else dx_m

                maps.append(
                    f'        <text x="{x:.2f}" y="{y:.2f}" dx="-{dx_mm:.2f}" dy="{dy_m:.2f}" '
                    f'text-anchor="end">{n}</text>'
                )
                mask["aam"].append(
                    f'            <text x="{x:.2f}" y="{y:.2f}" dx="-{dx_mm:.2f}" dy="{dy_m:.2f}" '
                    f'text-anchor="end">{n}</text>'
                )
        if others:
            svg.append(
                f'      <g font-family="{config["other_font_style"]}" fill="{other_fill}" '
                f'font-size="{other_size:.2f}">'
            )
            svg.extend(others)
            svg.append("      </g>")
        if mapping:
            svg.append(f'      <g fill="{map_fill}" font-size="{mapping_font:.2f}">')
            svg.extend(maps)
            svg.append("      </g>")
        return svg, mask


def depict_custom_reaction(reaction: ReactionContainer):
    """
    Depicts a ReactionContainer using custom atom rendering logic (replace At to X).

    This function generates an SVG string representing a reaction. It
    temporarily modifies the classes of the molecules within the reaction
    to use a custom depiction logic (`CustomDepictMolecule`) that alters
    how atoms are rendered (specifically, it seems to use `atom.symbol`
    instead of `atom.atomic_symbol`, potentially for replacing 'At' with 'X'
    as mentioned in the original comment). After depicting each molecule
    with the temporary class, it restores the original classes. The function
    then combines the individual molecule depictions, reaction arrow, and
    reaction signs into a single SVG.

    Args:
        reaction (ReactionContainer): The ReactionContainer object to be depicted.

    Returns:
        str: An SVG string representing the depiction of the reaction
             with custom atom rendering.
    """
    if not reaction._arrow:
        reaction.fix_positions()  # Ensure positions are calculated

    r_atoms = []
    r_bonds = []
    r_masks = []
    r_max_x = r_max_y = r_min_y = 0
    original_classes = {}  # Store original classes to restore later

    try:
        # Temporarily change the class of molecules to use the custom depiction
        for mol in reaction.molecules():
            if isinstance(mol, (MoleculeContainer, CGRContainer)):
                original_classes[mol] = mol.__class__
                custom_class_name = (
                    f"TempCustom_{mol.__class__.__name__}_{uuid4().hex}"  # Unique name
                )
                # Combine custom depiction with original class methods
                # Ensure the custom _render_atoms takes precedence
                new_bases = (CustomDepictMolecule,) + original_classes[mol].__bases__
                # Filter out DepictMolecule if it's already a base to avoid MRO issues
                new_bases = tuple(b for b in new_bases if b is not DepictMolecule)
                # If DepictMolecule wasn't a direct base, ensure its methods are accessible
                if CustomDepictMolecule not in original_classes[mol].__mro__:
                    # Prioritize CustomDepictMolecule's methods
                    new_bases = (CustomDepictMolecule, original_classes[mol])
                else:
                    # If DepictMolecule was a base, CustomDepictMolecule is already first
                    new_bases = (CustomDepictMolecule,) + tuple(
                        b
                        for b in original_classes[mol].__bases__
                        if b is not DepictMolecule
                    )

                # Create the temporary class
                mol.__class__ = type(custom_class_name, new_bases, {})

            # Depict using the (potentially) modified class
            atoms, bonds, masks, min_x, min_y, max_x, max_y = mol.depict(embedding=True)
            r_atoms.append(atoms)
            r_bonds.append(bonds)
            r_masks.append(masks)
            if max_x > r_max_x:
                r_max_x = max_x
            if max_y > r_max_y:
                r_max_y = max_y
            if min_y < r_min_y:
                r_min_y = min_y

    finally:
        # Restore original classes
        for mol, original_class in original_classes.items():
            mol.__class__ = original_class

    config = DepictMolecule._render_config  # Access via the imported class

    font_size = config["font_size"]
    font125 = 1.25 * font_size
    width = r_max_x + 3.0 * font_size
    height = r_max_y - r_min_y + 2.5 * font_size
    viewbox_x = -font125
    viewbox_y = -r_max_y - font125

    svg = [
        f'<svg width="{width:.2f}cm" height="{height:.2f}cm" '
        f'viewBox="{viewbox_x:.2f} {viewbox_y:.2f} {width:.2f} '
        f'{height:.2f}" xmlns="http://www.w3.org/2000/svg" version="1.1">\n'
        '  <defs>\n    <marker id="arrow" markerWidth="10" markerHeight="10" '
        'refX="0" refY="3" orient="auto">\n      <path d="M0,0 L0,6 L9,3"/>\n    </marker>\n  </defs>\n'
        f'  <line x1="{reaction._arrow[0]:.2f}" y1="0" x2="{reaction._arrow[1]:.2f}" y2="0" '
        'fill="none" stroke="black" stroke-width=".04" marker-end="url(#arrow)"/>'
    ]

    sings_plus = reaction._signs
    if sings_plus:
        svg.append('  <g fill="none" stroke="black" stroke-width=".04">')
        for x in sings_plus:
            svg.append(
                f'    <line x1="{x + .35:.2f}" y1="0" x2="{x + .65:.2f}" y2="0"/>'
            )
            svg.append(
                f'    <line x1="{x + .5:.2f}" y1="0.15" x2="{x + .5:.2f}" y2="-0.15"/>'
            )
        svg.append("  </g>")

    for atoms, bonds, masks in zip(r_atoms, r_bonds, r_masks):
        # Use the static method from Depict directly
        svg.extend(
            Depict._graph_svg(atoms, bonds, masks, viewbox_x, viewbox_y, width, height)
        )
    svg.append("</svg>")
    return "\n".join(svg)


def remove_and_shift(nested_dict, to_remove):  # Under development
    """
    Removes specified inner keys from a nested dictionary and renumbers the remaining keys.

    Given a dictionary where values are themselves dictionaries, this function
    iterates through each inner dictionary. For each inner dictionary, it
    creates a new dictionary containing only the key-value pairs where the
    inner key is NOT present in the `to_remove` list. The keys of the remaining
    elements in the new inner dictionary are then renumbered sequentially
    starting from 0, effectively removing gaps left by the removed keys.

    Args:
        nested_dict (dict): The input nested dictionary (dict of dicts).
        to_remove (list): A list of keys to remove from the inner dictionaries.

    Returns:
        dict: A new nested dictionary with the specified keys removed from
              inner dictionaries and the remaining inner keys renumbered.
    """
    rem_set = set(to_remove)

    result = {}
    for outer_k, inner in nested_dict.items():
        new_inner = {}
        for old_k, v in inner.items():
            if old_k in rem_set:
                continue
            shift = sum(1 for r in rem_set if r < old_k)
            new_k = old_k - shift
            new_inner[new_k] = v
        result[outer_k] = new_inner
    return result
