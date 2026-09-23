// countEvents.C

#include <TFile.h>
#include <TSystem.h>
#include <TSystemDirectory.h>
#include <TSystemFile.h>
#include <TTree.h>

#include <iostream>
#include <string>

Long64_t countEventsInDirectory(const std::string& directory,
                                Long64_t& nFiles,
                                Long64_t& nSkipped)
{
  Long64_t totalEvents = 0;

  TSystemDirectory currentDirectory("currentDirectory", directory.c_str());
  TList* entries = currentDirectory.GetListOfFiles();

  if (!entries) {
    std::cerr << "Could not read directory: " << directory << '\n';
    return 0;
  }

  TIter next(entries);
  while (auto* entry = dynamic_cast<TSystemFile*>(next())) {
    const std::string name = entry->GetName();

    if (name == "." || name == "..") {
      continue;
    }

    const std::string path = directory + "/" + name;

    if (entry->IsDirectory()) {
      totalEvents += countEventsInDirectory(path, nFiles, nSkipped);
      continue;
    }

    if (name.size() < 5 || name.substr(name.size() - 5) != ".root") {
      continue;
    }

    TFile file(path.c_str(), "READ");

    if (file.IsZombie()) {
      std::cerr << "Could not open: " << path << '\n';
      ++nSkipped;
      continue;
    }

    auto* eventTree = file.Get<TTree>("eventTree");

    if (!eventTree) {
      std::cerr << "No eventTree in: " << path << '\n';
      ++nSkipped;
      continue;
    }

    const Long64_t nEvents = eventTree->GetEntries();

    std::cout << path << ": " << nEvents << " events\n";

    totalEvents += nEvents;
    ++nFiles;
  }

  return totalEvents;
}

void countEvents()
{
  const std::string baseDirectory =
      "/rstorage/alice/run3/data/LHC25ae_mb_100/BerkeleyTrees";

  Long64_t nFiles = 0;
  Long64_t nSkipped = 0;

  const Long64_t totalEvents =
      countEventsInDirectory(baseDirectory, nFiles, nSkipped);

  std::cout << "\n========================================\n"
            << "ROOT files successfully processed: " << nFiles << '\n'
            << "Files skipped:                     " << nSkipped << '\n'
            << "Total number of events:            " << totalEvents << '\n'
            << "========================================\n";
}