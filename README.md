# Students.Filters

## Introduction

Students.Filters is a [package](http://weka.wikispaces.com/Packages) that provides [unsupervised learning](http://en.wikipedia.org/wiki/Unsupervised_learning) filters for the [WEKA](http://www.cs.waikato.ac.nz/~ml/weka/index.html) machine learning toolkit version >3.7. Development will prioritize filters that are useful to students taking machine learning at Georgia Tech; initially only an [Independent Component Analysis](http://en.wikipedia.org/wiki/Independent_component_analysis) filter using the [FastICA](http://research.ics.aalto.fi/ica/newindex.shtml) algorithm has been implemented.

## Installation

The preferred installation method is to use the WEKA package manager. The git repository contains additional files for an Eclipse project with Maven dependencies for the [EJML](https://code.google.com/p/efficient-java-matrix-library/) package, and Ant build files for the `jar`.

### WEKA Package Manager

See instructions on the WEKA [homepage](http://weka.wikispaces.com/How+do+I+use+the+package+manager%3F). If the package is not available from the official package page, it can be installed directly from: 

    https://github.com/cgearhart/students-filters/raw/master/StudentFilters.zip

### Git Repository

The source code & package file can be intalled from git:

    git clone https://github.com/cgearhart/students-filters.git

## Use

The filter can be used like other WEKA filters from the command line, from the WEKA GUI, or directly within your own Java code. The specific options for each file can be found in the source code, documentation, or from the command line with the `-h` flag.

### Command Line

Read the [instructions](http://weka.wikispaces.com/How+do+I+use+WEKA+from+command+line%3F) first. Make sure that `weka.jar` and the `StudentFilters.jar` files are in the classpath and in order. Options for each filter can be determined with the `-h` argument. The filter can then be directly invoked (or chained like other WEKA filters), e.g.:

    java -cp <weka_path>/weka.jar:<weka_packages>/studentfilters.jar weka.filters.unsupervised.attribute.IndependentComponent -i <infile.arff> -o <outfile.arff> -W -A -1 -N 200 -T 1E-4

### IDE

The FastICA algorithm is implemented indepdent of WEKA, so it can be included without adding WEKA to your project by including the `StudentFilters.jar` file and importing `filters.FastICA`. However, using the WEKA-compatible IndepdentComponents filter  requires the `weka.jar` in the classpath, and can be imported as `weka.filters.unsupervised.attribute.IndependentComponents`. See the WEKA [documentation](http://weka.wikispaces.com/Use+WEKA+in+your+Java+code) for more details.

The `build.xml` file can be used with [Apache Ant](http://ant.apache.org/) to rebuild `StudentFilters.jar` by running:

    ant build

NOTE: the EMJL library needs to be installed on your system in the expected location; follow the [instructions](https://code.google.com/p/efficient-java-matrix-library/) to install it with [Maven](http://maven.apache.org/).

Sample usage:
```
		Instances convertedInstances = null;
		if (requiresNominalToBinaryFilter) {
			NominalToBinary nominalToBinaryFilter = new NominalToBinary();
			nominalToBinaryFilter.setInputFormat(inputDataSet);
			convertedInstances = Filter.useFilter(inputDataSet, nominalToBinaryFilter);
		} else {
			convertedInstances = inputDataSet;
		}

		Remove removeFilter = new Remove();
		removeFilter.setAttributeIndices("" + (convertedInstances.classIndex() + 1));
		removeFilter.setInputFormat(convertedInstances);
		Instances filteredInstances = Filter.useFilter(convertedInstances, removeFilter);

		IndependentComponents ica = new IndependentComponents();
		ica.setNumAttributes(10); // change
		ica.setInputFormat(filteredInstances);
		Instances transformedInstances = Filter.useFilter(filteredInstances, ica);
```


### WEKA GUI

Once the filter is installed with the package manager, or has been simply unzipped to the package folder on the weka path, it will automatically appear in the WEKA gui. (The GUI must usually be restarted after new packages are added.) See the WEKA [documentation](http://weka.wikispaces.com/How+do+I+use+the+package+manager%3F) for more details.

### Alternative Maven Build

The `pom.xml` file can be used with [Apache Maven](http://maven.apache.org/) to rebuild `filters-0.0.1-SNAPSHOT.jar` by running:

    mvn install -Dmaven.test.skip=true

NOTE: dependencies will be handled automatically by Maven.

GUI can then be launched with

	java -Xmx1g -classpath <maven_path>/.m2/repository/com/googlecode/efficient-java-matrix-library/ejml/0.25/ejml-0.25.jar:<maven_path>/.m2/repository/nz/ac/waikato/cms/weka/weka-dev/3.7.10/weka-dev-3.7.10.jar:<maven_path>/.m2/repository/net/sf/squirrel-sql/thirdparty-non-maven/java-cup/0.11a/java-cup-0.11a.jar:<maven_path>/.m2/repository/org/pentaho/pentaho-commons/pentaho-package-manager/1.0.8/pentaho-package-manager-1.0.8.jar:<maven_path>/.m2/repository/junit/junit/4.11/junit-4.11.jar:<maven_path>/.m2/repository/org/hamcrest/hamcrest-core/1.3/hamcrest-core-1.3.jar weka.gui.Main

## License

The filters are dependent on [WEKA](http://www.cs.waikato.ac.nz/~ml/weka/index.html) (licensed under [GPL](http://www.gnu.org/licenses/gpl.html)) and the Efficient Java Matrix Library ([EJML](https://code.google.com/p/efficient-java-matrix-library/)) (licensed under [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)). The [FastICA](http://research.ics.aalto.fi/ica/newindex.shtml) algorithm is released under the [GPL](http://research.ics.aalto.fi/ica/fastica/about.shtml). The implementation in this package is based on the [scikit-learn](http://scikit-learn.org/stable/index.html) implementation which is released under [BSD](https://github.com/scikit-learn/scikit-learn/blob/master/COPYING). To the extent that there may be any original copyright, it is licensed under the [Unlicense](http://unlicense.org/) - i.e., it is released to the Public Domain.