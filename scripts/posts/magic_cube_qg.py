"""Script for making a slowly rotating XKCD style cube with text at the vertices,
showing how various physical constants are involved in the construction of theories
"""

import matplotlib.pyplot as plt

VERTICES = [
    (0, 0, 0),  # None - Galilean physics
    (1, 0, 0),  # G - Newtonian physics, Newtonian gravity (curvature Newton-Cartan)
    (0, 1, 0),  # c - classical field theory, Electromagnetism (Maxwell), special relativity
    (0, 0, 1),  # hbar - Quantum Mechanics (non relativistic), Planck constant, discrete spectra of observables
    (1, 1, 0),  # G + c - General Relativity (Newton + Maxwell)
    (1, 0, 1),  # G + hbar - N/A
    (0, 1, 1),  # c + hbar - Quantum Field Theory (Quantum Electrodynamics (QED))
    (1, 1, 1),  # G + c + hbar - Quantum Gravity (General Relativity + Quantum Mechanics)
]


def plot_magic_cube_static():
    """

    Returns:

    """
    # Set the style to XKCD
    plt.xkcd()

    # Hide the plot borders
    plt.axis('off')

    # Setup a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Set the axis limits
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_zlim(-0.1, 1.1)

    # Hide gray background
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # Plot the edges of the cube as solid lines
    for i in range(len(VERTICES)):
        for j in range(len(VERTICES)):
            if i < j:
                # Check v(i) and v(j) share at two components
                if sum([1 for k in range(3) if VERTICES[i][k] == VERTICES[j][k]]) == 2:
                    ax.plot(*zip(VERTICES[i], VERTICES[j]), color='black')

    # Add axis labels
    # ax.zaxis.set_rotate_label(False)
    # ax.set_xlabel('G')
    # ax.set_ylabel('c')
    # ax.set_zlabel('$\hbar$')

    # Hide the axes lines
    ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

    # Disable axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # Move axis labels closer to the vertices
    ax.xaxis.labelpad = -10
    ax.yaxis.labelpad = -10
    ax.zaxis.labelpad = -10

    # Add title
    plt.title('" Magic Cube " of Theoretical Physics')

    # Add text labels to the vertices
    ax.text(-0.2, -0.45, -0.1, 'Galilean\nPhysics', color='black')
    ax.text(1.22, -0.3, -0.1, 'Newtonian\nGravity', color='blue')
    ax.text(-0.5, 0.7, 0, 'Special\nRelativity', color='red')
    ax.text(-0.35, -0.3, 1.075, 'Quantum\nMechanics', color='green')
    ax.text(1.07, 1.0, -0.25, 'General\nRelativity', color='purple')
    ax.text(1, -0.03, 1.08, '?', color='grey')
    ax.text(-0.1, 1.1, 1, 'Quantum\nField Theory', color='brown')
    ax.text(1.0, 1.09, 0.98, 'Quantum\nGravity', color='black')

    # Redraw the edges to match colors of dimensions (G is blue, c is red, hbar is green)
    ax.plot([0, 1], [0, 0], [0, 0], color='blue')
    ax.plot([0, 0], [0, 1], [0, 0], color='red')
    ax.plot([0, 0], [0, 0], [0, 1], color='green')
    ax.plot([1, 1], [0, 1], [0, 0], color='black')
    ax.plot([1, 0], [1, 1], [0, 0], color='black')
    ax.plot([0, 0], [1, 1], [0, 1], color='black')
    ax.plot([1, 1], [1, 1], [0, 1], color='black')
    ax.plot([1, 0], [1, 1], [1, 1], color='black')
    ax.plot([1, 1], [0, 1], [1, 1], color='grey')
    ax.plot([1, 1], [0, 0], [0, 1], color='grey')
    ax.plot([0, 0], [0, 1], [1, 1], color='black')
    ax.plot([1, 0], [0, 0], [1, 1], color='greym')

    # Redraw the vertices to match colors of text
    weight = 3
    ax.scatter(0, 0, 0, linewidths=weight, color='black')
    ax.scatter(1, 0, 0, linewidths=weight, color='blue')
    ax.scatter(0, 1, 0, linewidths=weight, color='red')
    ax.scatter(0, 0, 1, linewidths=weight, color='green')
    ax.scatter(1, 1, 0, linewidths=weight, color='purple')
    # ax.scatter(1, 0, 1, linewidths=weight, color='white')
    ax.scatter(0, 1, 1, linewidths=weight, color='brown')
    ax.scatter(1, 1, 1, linewidths=weight, color='black')

    # Draw arrow marks in middle of vertices to indicate positive direction
    ax.plot([0.45, 0.5, 0.4], [-0.05, 0, 0.05], [0, 0, 0], color='blue')
    ax.plot([0, 0, 0], [0.45, 0.5, 0.4], [-0.05, 0, 0.05], color='red')
    ax.plot([-0.04, 0, 0.035], [0, 0, 0], [0.38, 0.5, 0.4], color='green')

    # Add text labels for axes
    ax.text(0.35, 0, -0.4, '$G$', color='blue', size=28)
    ax.text(0.15, 0.45, -0.1, '$c$', color='red', size=28)
    ax.text(-0.3, 0.0, 0.28, '$\hbar$', color='green', size=28)

    # Set the viewing angle
    ax.view_init(elev=22.5, azim=-52.5)

    # Tight layout
    plt.tight_layout()

    # Save the plot
    plt.savefig('magic_cube.png', dpi=300)

    # Show the plot
    plt.show()


def main():
    """Main function"""
    plot_magic_cube_static()


if __name__ == '__main__':
    main()
