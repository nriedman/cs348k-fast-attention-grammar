from renderer.renderer import render
from grammar.atomic import attention_atomic_grammar

def main():
    g = attention_atomic_grammar()
    output_path = "kernels/test.py"
    render(g, output_path=output_path)

if __name__ == "__main__":
    main()