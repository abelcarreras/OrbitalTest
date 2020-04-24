def get_2nd_order_overlap_matrix(basis, basis2):

    g1, g2 = basis.get_ao_functions()
    g1b, g2b = basis2.get_ao_functions()

    overlap_2nd_order = [[(g1 * g1 * g1b * g1b).full_integration(), (g1 * g1 * g1b * g2b).full_integration(), (g1 * g2 * g1b * g1b).full_integration(), (g1 * g2 * g1b * g2b).full_integration()],
                         [(g1 * g1 * g2b * g1b).full_integration(), (g1 * g1 * g2b * g2b).full_integration(), (g1 * g2 * g2b * g1b).full_integration(), (g1 * g2 * g2b * g2b).full_integration()],
                         [(g2 * g1 * g1b * g1b).full_integration(), (g2 * g1 * g1b * g2b).full_integration(), (g2 * g2 * g1b * g1b).full_integration(), (g2 * g2 * g1b * g2b).full_integration()],
                         [(g2 * g1 * g2b * g1b).full_integration(), (g2 * g1 * g2b * g2b).full_integration(), (g2 * g2 * g2b * g1b).full_integration(), (g2 * g2 * g2b * g2b).full_integration()]]

    return overlap_2nd_order
