import numpy as np
from numpy import genfromtxt
import os

def load_file(folder, file_name):
    return genfromtxt(folder + file_name, delimiter=',', dtype='str').transpose()

def load_data():
    folder = '../Data/'

    # Load the files into string matrices
    gdsc_dr = load_file(folder, 'gdsc_dr_lnIC50.csv')
    tcga_dr = load_file(folder, 'tcga_dr.csv')
    gdsc_expr = load_file(folder, 'gdsc_expr_postCB.csv')
    tcga_expr = load_file(folder, 'tcga_expr_postCB.csv')

    # Verify that gdsc_dr/tcga_dr drugs are in same order
    try:
        for col in range (1, np.size(gdsc_dr, 1)):
            assert gdsc_dr[0, col] == tcga_dr[0, col]
    except AssertionError:
        print("AssertionError: Drug names do not match up")

    # Verify that gdsc_expr/tcga_expr ensemble gene ids are in same order
    try:
        for col in range (1, np.size(gdsc_expr, 1)):
            assert gdsc_expr[0, col] == tcga_expr[0, col]
    except AssertionError:
        print("AssertionError: Ensemble gene ids do not match up")

    # Verify that gdsc_dr/gdsc_expr cell line ids are in same order
    try:
        for row in range (1, np.size(gdsc_dr, 0)):
            assert gdsc_dr[row, 0] == gdsc_expr[row, 0]
    except AssertionError:
        print("AssertionError: Cell line ids do not match up")

    # Verify that tcga_dr/tcga_expr patient ids are in same order
    try:
        for row in range (1, np.size(tcga_dr, 0)):
            assert tcga_dr[row, 0] == tcga_expr[row, 0]
    except AssertionError:
        print("AssertionError: Patient ids do not match up")

    # Remove the row and column names
    gdsc_dr = gdsc_dr[1:, 1:]
    tcga_dr = tcga_dr[1:, 1:]
    gdsc_expr = gdsc_expr[1:, 1:]
    tcga_expr = tcga_expr[1:, 1:]

    # Save the matrices locally
    np.save('gdsc_dr.npy', gdsc_dr)
    np.save('tcga_dr.npy', tcga_dr)
    np.save('gdsc_expr.npy', gdsc_expr)
    np.save('tcga_expr.npy', tcga_expr)

load_data()