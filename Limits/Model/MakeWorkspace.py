import ROOT

# Load Data and Signal
f = ROOT.TFile("../Model/shapes_X300.root", "READ") # I exported the data and original MC to this file. 
data = f.Get("data_obs")
signal = f.Get("signal_m75")

# Create a RooFit workspace and variables
w = ROOT.RooWorkspace("w")

# Define observable (x-axis of the histogram)
mass = ROOT.RooRealVar("x", "x", 180, 1000)
mass.setMin(180)

#Make datahists for signal and data
signal_hist = ROOT.RooDataHist("signal_mc", "signal_mc", ROOT.RooArgList(mass), signal)
data_hist = ROOT.RooDataHist("data_obs", "data_obs", ROOT.RooArgList(mass), data)



# Define Diphoton Parameters for Background
p0 = ROOT.RooRealVar("p0", "Coefficient", 1.00000e-02, 1e-4, 100)   
p1 = ROOT.RooRealVar("p1", "Exponent Base", 5.98255e+00, -10, 10)
p2 = ROOT.RooRealVar("p2", "Logarithmic Term", -8.11258e-01, -10, 10)



# Create the Diphoton PDF
# Define the background model: p0 * x^(p1 + p2 * log(x))
bkg_expr = ROOT.RooGenericPdf(
    "bkg_model", "p0 * pow(x, (p1 + p2 * log(x)))", ROOT.RooArgList(mass, p0, p1, p2)
)

# Set parameters based on background fit
bkg_expr.fitTo(data_hist)
# Define the integration range
mass.setRange("fullRange", 180, 1000)

# Compute the integral of the background model over the given range
bkg_integral = bkg_expr.createIntegral(ROOT.RooArgSet(mass), ROOT.RooFit.NormSet(ROOT.RooArgSet(mass)), ROOT.RooFit.Range("fullRange"))

# Get the total yield by multiplying by the number of events in the dataset
bkg_yield = bkg_integral.getVal() * data_hist.sumEntries()

# Print the integral value
print("Background yield over the mass range: ", bkg_yield)


# Compute the total number of events in the specified mass range
data_yield = data_hist.sumEntries()

# Print the total yield of data in the mass range
print("Total data yield over the mass range: ", data_yield)
# Let's plot the model fit to the data
can = ROOT.TCanvas()
plot = mass.frame()
data_hist.plotOn(plot)
bkg_expr.plotOn(plot, ROOT.RooFit.LineColor(2))
plot.Draw()
can.Update()
can.Draw()
can.SaveAs("bkg_model-fit.png")

mass.setRange(180, 1000)

# Define parameters for the double-sided Crystal Ball function (signal)
mean = ROOT.RooRealVar("mean", "mean", 300, 280, 320)
sigma = ROOT.RooRealVar("sigma", "sigma", 10, 1, 15)
alpha1 = ROOT.RooRealVar("alpha1", "alpha1", 1.5, 0.1, 5)
n1 = ROOT.RooRealVar("n1", "n1", 2.0, 0.1, 5)
alpha2 = ROOT.RooRealVar("alpha2", "alpha2", 1.5, 0.1, 5)
n2 = ROOT.RooRealVar("n2", "n2", 2.0, 0.1, 5)

# Create the double-sided Crystal Ball PDF
cb_pdf = ROOT.RooCrystalBall("cb_pdf", "cb_pdf", mass, mean, sigma, alpha1, n1, alpha2, n2)

# Set parameters based on DCB fit
cb_pdf.fitTo(signal_hist,ROOT.RooFit.SumW2Error(True))

can = ROOT.TCanvas()
plot = mass.frame()
signal_hist.plotOn(plot)
cb_pdf.plotOn( plot, ROOT.RooFit.LineColor(2) )
plot.Draw()
can.Update()
can.Draw()
can.SaveAs("signal_model-fit.png")

# Set Signal params constant
mean.setConstant(True)
sigma.setConstant(True)
alpha1.setConstant(True)
n1.setConstant(True)
alpha2.setConstant(True)
n2.setConstant(True)





# Let bkg float #
p0.setConstant(False)
p1.setConstant(False)
p2.setConstant(False)

# Import to Workspace #
getattr(w, "import")(data_hist)
getattr(w, "import")(cb_pdf)
getattr(w, "import")(bkg_expr)

# Save workspace
w.Print()
w.writeToFile("workspace_m75.root")
