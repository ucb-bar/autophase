#include "llvm/Pass.h"
#include "llvm/IR/Module.h"
//#include "llvm/Support/raw_ostream.h"
#include "llvm/Analysis/Passes.h"
//#include "llvm/Analysis/PassSupport.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;
#define DEBUG_TYPE "globalvarpass"

STATISTIC(NumGlobalVars, "Number of Global Variables referred to (number present)");
STATISTIC(GlobalVarUsage, "Number of Global Variable References (usages)");

namespace {
  class GlobalVar : public ModulePass {
  
  public: 
    static char ID;
    GlobalVar() : ModulePass(ID) {
       initializeGlobalVarPass(*PassRegistry::getPassRegistry());
    }

  bool runOnModule(Module &M) override {
	for (auto &Global : M.getGlobalList()) {
		//errs() << "Testing: ";
		//errs().write_escaped(M.getModuleIdentifier()) << '\n';

		NumGlobalVars++;
		if (auto *v = dyn_cast<Value>(&Global)) {
			GlobalVarUsage += v->getNumUses();
		}
	}

    return false;
  }
}; 
}

char GlobalVar::ID = 0;
INITIALIZE_PASS(GlobalVar, "globalvarpass",
                "Counts the various types of Instructions", false, true)

ModulePass *llvm::createGlobalVarPass() { 
  return new GlobalVar(); 
}

